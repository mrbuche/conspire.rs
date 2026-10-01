#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef, DirectionsRef,
        bbox::BoundingBox,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::{Quantity, Scalar, Tensor},
};
use std::{array::from_fn, sync::LazyLock};

type Point = [Scalar; 3];
type Weights = [[Scalar; 8]; 3];

const ARMIJO: Scalar = 1.0e-4;
const CAP: usize = 4;
const CREASE: Scalar = 0.866;
const CURVATURE_FLOOR: Scalar = 1.0e-12;
const HISTORY: usize = 10;
const ITERATIONS: usize = 50;
const MINIMUM_STEP: Scalar = 1.0e-8;
const RADIUS: Scalar = 0.2;
const RHO: Scalar = 1.0e2;
const STALL: Scalar = 1.0e-6;
const STEP: Scalar = 0.01;
const SWEEPS: usize = 20;
const TOLERANCE: Scalar = 1.0e-3;

static POINTS: LazyLock<[Weights; 9]> = LazyLock::new(|| {
    let corners = [
        (0, [1, 3, 4]),
        (1, [2, 0, 5]),
        (2, [3, 1, 6]),
        (3, [0, 2, 7]),
        (4, [7, 5, 0]),
        (5, [4, 6, 1]),
        (6, [5, 7, 2]),
        (7, [6, 4, 3]),
    ];
    let faces = [
        ([1, 2, 5, 6], [0, 3, 4, 7]),
        ([2, 3, 6, 7], [0, 1, 4, 5]),
        ([4, 5, 6, 7], [0, 1, 2, 3]),
    ];
    from_fn(|point| {
        let mut weights = [[0.0; 8]; 3];
        if point < 8 {
            let (corner, adjacent) = corners[point];
            (0..3).for_each(|i| {
                weights[i][adjacent[i]] += 1.0;
                weights[i][corner] -= 1.0;
            })
        } else {
            (0..3).for_each(|i| {
                faces[i].0.iter().for_each(|&node| weights[i][node] += 0.25);
                faces[i].1.iter().for_each(|&node| weights[i][node] -= 0.25)
            })
        }
        weights
    })
});

struct Plane {
    boundary: usize,
    offset: Scalar,
    normal: Point,
    slot: usize,
}

struct Energy<'a> {
    hexes: &'a [[usize; 8]],
    lambda: &'a [[Scalar; CAP]],
    planes: &'a [Plane],
    rho: Scalar,
    boundary: &'a [usize],
    scales: &'a [Scalar],
    theta: Scalar,
}

impl Mesh<3> {
    pub fn improve_scaled_jacobian(&mut self, target: &Tessellation) -> Result<(), &'static str> {
        let mut hexes = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(elements) => hexes.extend(elements.iter().copied()),
                _ => return Err("improve_scaled_jacobian requires hexahedra"),
            }
        }
        let mut boundary: Vec<usize> = self.exterior_faces().into_iter().flatten().collect();
        boundary.sort_unstable();
        boundary.dedup();
        let coordinates = self.coordinates.members_mut();
        let mut x: Vec<Scalar> = coordinates
            .iter()
            .flat_map(|node| (0..3).map(|i| node[i].value()))
            .collect();
        let surface = Surface::new(target);
        let mut lambda = vec![[0.0; CAP]; boundary.len()];
        let mut rho = RHO;
        let mut best: Option<Vec<Scalar>> = None;
        let mut theta = 0.0;
        let mut first = true;
        'ratchet: loop {
            for _ in 0..SWEEPS {
                let scales = scales(&hexes, &x);
                let planes = surface.planes(&boundary, &hexes, &scales, &x);
                let quality = minimum_scaled_jacobian(&hexes, &x);
                let gap = gap(&planes, &boundary, &hexes, &scales, &x);
                if first {
                    first = false;
                    if gap <= TOLERANCE {
                        best = Some(x.clone());
                        theta = ((quality / STEP).floor() + 1.0) * STEP;
                    }
                }
                if quality >= theta && gap <= TOLERANCE {
                    best = Some(x.clone());
                    theta = ((quality / STEP).floor() + 1.0) * STEP;
                    rho *= 2.0;
                    continue 'ratchet;
                }
                let energy = Energy {
                    hexes: &hexes,
                    lambda: &lambda,
                    planes: &planes,
                    rho,
                    boundary: &boundary,
                    scales: &scales,
                    theta,
                };
                let decrease = energy.minimize(&mut x);
                planes.iter().for_each(|plane| {
                    lambda[plane.boundary][plane.slot] += rho * plane.residual(&x, &boundary)
                });
                if decrease < STALL {
                    break;
                }
            }
            break;
        }
        let x = best.ok_or("could not fit the surface")?;
        coordinates
            .iter_mut()
            .enumerate()
            .for_each(|(node, entry)| {
                (0..3).for_each(|i| entry[i] = Quantity::new(x[3 * node + i]))
            });
        Ok(())
    }
}

impl Plane {
    fn residual(&self, x: &[Scalar], boundary: &[usize]) -> Scalar {
        let node = boundary[self.boundary];
        (0..3)
            .map(|i| self.normal[i] * x[3 * node + i])
            .sum::<Scalar>()
            - self.offset
    }
}

struct Surface<'a> {
    normals: DirectionsRef<'a, 3>,
    surface: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    target: &'a Tessellation,
}

impl<'a> Surface<'a> {
    fn new(target: &'a Tessellation) -> Self {
        let mesh = target.mesh();
        Self {
            normals: target.normals().iter().flatten().collect(),
            surface: mesh.coordinates(),
            elements: mesh.connectivities().iter().flatten().collect(),
            target,
        }
    }
    fn planes(
        &self,
        boundary: &[usize],
        hexes: &[[usize; 8]],
        scales: &[Scalar],
        x: &[Scalar],
    ) -> Vec<Plane> {
        let sizes = node_scales(boundary, hexes, scales, x.len() / 3);
        let bvh = self.target.bvh();
        let mut planes = Vec::new();
        boundary.iter().enumerate().for_each(|(index, &node)| {
            let position = Coordinate::<3>::from(from_fn(|i| x[3 * node + i]));
            let Some((point, first)) = bvh.closest_point(&position, self.surface, &self.elements)
            else {
                return;
            };
            let reach = RADIUS * sizes[node];
            let (low, high) = (
                Coordinate::<3>::from(from_fn(|i| point[i].value() - reach)),
                Coordinate::<3>::from(from_fn(|i| point[i].value() + reach)),
            );
            let query: BoundingBox<3> = [&low, &high]
                .into_iter()
                .collect::<CoordinatesRef<'_, 3>>()
                .into();
            let mut kept: Vec<Point> = Vec::new();
            for triangle in std::iter::once(first).chain(bvh.overlapping(&query)) {
                if kept.len() == CAP {
                    break;
                }
                let nodes = self.elements[triangle];
                if triangle != first
                    && !query.overlaps_triangle(
                        &self.surface[nodes[0]],
                        &self.surface[nodes[1]],
                        &self.surface[nodes[2]],
                    )
                {
                    continue;
                }
                let normal: Point = from_fn(|i| self.normals[triangle][i].value());
                if kept.iter().any(|other| dot(other, &normal).abs() > CREASE) {
                    continue;
                }
                let vertex = &self.surface[self.elements[triangle][0]];
                planes.push(Plane {
                    boundary: index,
                    offset: (0..3).map(|i| normal[i] * vertex[i].value()).sum(),
                    normal,
                    slot: kept.len(),
                });
                kept.push(normal)
            }
        });
        planes
    }
}

impl Energy<'_> {
    fn value(&self, x: &[Scalar]) -> Scalar {
        self.evaluate(x, None)
    }
    fn evaluate(&self, x: &[Scalar], mut gradient: Option<&mut Vec<Scalar>>) -> Scalar {
        let mut total = 0.0;
        self.hexes
            .iter()
            .zip(self.scales)
            .for_each(|(hex, &scale)| {
                let nodes: [Point; 8] = from_fn(|n| from_fn(|i| x[3 * hex[n] + i]));
                let (value, slope) = terms(self.theta, scale, &nodes, gradient.is_some());
                total -= value;
                if let Some(gradient) = gradient.as_deref_mut() {
                    (0..8)
                        .for_each(|n| (0..3).for_each(|i| gradient[3 * hex[n] + i] -= slope[n][i]))
                }
            });
        self.planes.iter().for_each(|plane| {
            let residual = plane.residual(x, self.boundary);
            let multiplier = self.lambda[plane.boundary][plane.slot];
            total += multiplier * residual + 0.5 * self.rho * residual * residual;
            if let Some(gradient) = gradient.as_deref_mut() {
                let node = self.boundary[plane.boundary];
                (0..3).for_each(|i| {
                    gradient[3 * node + i] += (multiplier + self.rho * residual) * plane.normal[i]
                })
            }
        });
        total
    }
    fn minimize(&self, x: &mut [Scalar]) -> Scalar {
        let typical = 0.1 * self.scales.iter().sum::<Scalar>() / self.scales.len().max(1) as Scalar;
        let mut gradient = vec![0.0; x.len()];
        let mut value = self.evaluate(x, Some(&mut gradient));
        let initial = value;
        let mut history: Vec<(Vec<Scalar>, Vec<Scalar>)> = Vec::new();
        let mut trial = x.to_vec();
        for _ in 0..ITERATIONS {
            let peak = gradient
                .iter()
                .fold(0.0, |peak: Scalar, g| peak.max(g.abs()));
            if peak < 1.0e-14 {
                break;
            }
            let d = direction(&gradient, &history, typical / peak);
            let slope = dot_slice(&gradient, &d);
            if slope >= 0.0 {
                history.clear();
                continue;
            }
            let mut step = 1.0;
            let mut accepted = None;
            while step >= MINIMUM_STEP {
                trial
                    .iter_mut()
                    .zip(x.iter().zip(&d))
                    .for_each(|(t, (x, d))| *t = x + step * d);
                let candidate = self.value(&trial);
                if candidate <= value + ARMIJO * step * slope {
                    accepted = Some(candidate);
                    break;
                }
                step *= 0.5;
            }
            let Some(candidate) = accepted else {
                if history.is_empty() {
                    break;
                }
                history.clear();
                continue;
            };
            let s: Vec<Scalar> = d.iter().map(|d| d * step).collect();
            x.iter_mut().zip(&s).for_each(|(x, s)| *x += s);
            let mut updated = vec![0.0; x.len()];
            self.evaluate(x, Some(&mut updated));
            let y: Vec<Scalar> = updated.iter().zip(&gradient).map(|(n, o)| n - o).collect();
            if dot_slice(&s, &y) > CURVATURE_FLOOR * norm(&s) * norm(&y) {
                if history.len() == HISTORY {
                    history.remove(0);
                }
                history.push((s, y));
            }
            gradient = updated;
            value = candidate;
        }
        (initial - value) / initial.abs().max(Scalar::MIN_POSITIVE)
    }
}

fn terms(
    theta: Scalar,
    scale: Scalar,
    nodes: &[Point; 8],
    derivative: bool,
) -> (Scalar, [Point; 8]) {
    let mut lowest_j = (Scalar::INFINITY, 0);
    let mut lowest_sj = (Scalar::INFINITY, 0);
    let mut all = [[[0.0; 3]; 3]; 9];
    for (point, weights) in POINTS.iter().enumerate() {
        let e = vectors(weights, nodes);
        let j = triple(&e[0], &e[1], &e[2]);
        let lengths = e.map(|e| norm(&e));
        let sj = if lengths.iter().all(|&l| l > 0.0) {
            j / (lengths[0] * lengths[1] * lengths[2])
        } else {
            0.0
        };
        all[point] = e;
        if j < lowest_j.0 {
            lowest_j = (j, point)
        }
        if sj < lowest_sj.0 {
            lowest_sj = (sj, point)
        }
    }
    let mut slope = [[0.0; 3]; 8];
    if lowest_j.0 <= 0.0 {
        if derivative {
            let e = &all[lowest_j.1];
            let g = [
                cross(&e[1], &e[2]),
                cross(&e[2], &e[0]),
                cross(&e[0], &e[1]),
            ];
            scatter(&POINTS[lowest_j.1], &g, 1.0 / scale, &mut slope)
        }
        (lowest_j.0 / scale, slope)
    } else if lowest_sj.0 <= theta {
        if derivative {
            let e = &all[lowest_sj.1];
            let u = e.map(|e| {
                let l = norm(&e);
                [e[0] / l, e[1] / l, e[2] / l]
            });
            let g: [Point; 3] = from_fn(|i| {
                let c = cross(&u[(i + 1) % 3], &u[(i + 2) % 3]);
                let along = dot(&c, &u[i]);
                let l = norm(&e[i]);
                from_fn(|k| (c[k] - along * u[i][k]) / l)
            });
            scatter(&POINTS[lowest_sj.1], &g, scale * scale, &mut slope)
        }
        (lowest_sj.0 * scale * scale, slope)
    } else {
        (theta * scale * scale, slope)
    }
}

fn scatter(weights: &Weights, g: &[Point; 3], factor: Scalar, slope: &mut [Point; 8]) {
    (0..3).for_each(|i| {
        (0..8).for_each(|n| (0..3).for_each(|k| slope[n][k] += factor * weights[i][n] * g[i][k]))
    })
}

fn vectors(weights: &Weights, nodes: &[Point; 8]) -> [Point; 3] {
    from_fn(|i| from_fn(|k| (0..8).map(|n| weights[i][n] * nodes[n][k]).sum()))
}

fn minimum_scaled_jacobian(hexes: &[[usize; 8]], x: &[Scalar]) -> Scalar {
    hexes
        .iter()
        .map(|hex| {
            let nodes: [Point; 8] = from_fn(|n| from_fn(|i| x[3 * hex[n] + i]));
            POINTS
                .iter()
                .map(|weights| {
                    let e = vectors(weights, &nodes);
                    let l = e.map(|e| norm(&e));
                    if l.iter().all(|&l| l > 0.0) {
                        triple(&e[0], &e[1], &e[2]) / (l[0] * l[1] * l[2])
                    } else {
                        0.0
                    }
                })
                .fold(Scalar::INFINITY, Scalar::min)
        })
        .fold(Scalar::INFINITY, Scalar::min)
}

fn scales(hexes: &[[usize; 8]], x: &[Scalar]) -> Vec<Scalar> {
    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    hexes
        .iter()
        .map(|hex| {
            EDGES
                .iter()
                .map(|&[a, b]| {
                    (0..3)
                        .map(|i| (x[3 * hex[a] + i] - x[3 * hex[b] + i]).powi(2))
                        .sum::<Scalar>()
                        .sqrt()
                })
                .sum::<Scalar>()
                / 12.0
        })
        .collect()
}

fn node_scales(
    boundary: &[usize],
    hexes: &[[usize; 8]],
    scales: &[Scalar],
    nodes: usize,
) -> Vec<Scalar> {
    let mut sum = vec![0.0; nodes];
    let mut count: Vec<Scalar> = vec![0.0; nodes];
    hexes.iter().zip(scales).for_each(|(hex, &scale)| {
        hex.iter().for_each(|&node| {
            sum[node] += scale;
            count[node] += 1.0
        })
    });
    let mut sizes = vec![0.0; nodes];
    boundary
        .iter()
        .for_each(|&node| sizes[node] = sum[node] / count[node].max(1.0));
    sizes
}

fn gap(
    planes: &[Plane],
    boundary: &[usize],
    hexes: &[[usize; 8]],
    scales: &[Scalar],
    x: &[Scalar],
) -> Scalar {
    let sizes = node_scales(boundary, hexes, scales, x.len() / 3);
    planes
        .iter()
        .map(|plane| plane.residual(x, boundary).abs() / sizes[boundary[plane.boundary]])
        .fold(0.0, Scalar::max)
}

fn direction(
    gradient: &[Scalar],
    history: &[(Vec<Scalar>, Vec<Scalar>)],
    fallback: Scalar,
) -> Vec<Scalar> {
    let mut q = gradient.to_vec();
    let mut alphas = vec![0.0; history.len()];
    let rhos: Vec<Scalar> = history.iter().map(|(s, y)| 1.0 / dot_slice(y, s)).collect();
    history.iter().enumerate().rev().for_each(|(k, (s, y))| {
        alphas[k] = rhos[k] * dot_slice(s, &q);
        q.iter_mut().zip(y).for_each(|(q, y)| *q -= alphas[k] * y)
    });
    let gamma = history
        .last()
        .map_or(fallback, |(s, y)| dot_slice(s, y) / dot_slice(y, y));
    q.iter_mut().for_each(|q| *q *= gamma);
    history.iter().enumerate().for_each(|(k, (s, y))| {
        let beta = rhos[k] * dot_slice(y, &q);
        q.iter_mut()
            .zip(s)
            .for_each(|(q, s)| *q += (alphas[k] - beta) * s)
    });
    q.iter_mut().for_each(|q| *q = -*q);
    q
}

fn dot(a: &Point, b: &Point) -> Scalar {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot_slice(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn norm(a: &[Scalar]) -> Scalar {
    a.iter().map(|a| a * a).sum::<Scalar>().sqrt()
}

fn cross(a: &Point, b: &Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn triple(a: &Point, b: &Point, c: &Point) -> Scalar {
    dot(a, &cross(b, c))
}
