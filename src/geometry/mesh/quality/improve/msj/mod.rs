#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates, CoordinatesRef, Direction, DirectionList,
        DirectionsRef,
        bbox::BoundingBox,
        mesh::{
            Connectivity, Mesh, Tessellation,
            lbfgs::{Objective, Settings, minimize},
        },
    },
    math::{
        CrossProduct, Quantity, QuantityVector, Reference, Scalar, Tensor, TensorList,
        TensorListVec, TensorRank1, TensorRank1Vec,
    },
    units::{Area, Dimensionless, Length, ReciprocalLength, Volume},
};
use std::{array::from_fn, iter::once, sync::LazyLock};

type Gradient = TensorRank1Vec<3, Reference, Length>;
type Weights = [[Scalar; 8]; 3];

const CAP: usize = 4;
const CREASE: Quantity = Dimensionless::of(0.866);
const FIRST: Quantity = Dimensionless::of(0.01);
const RADIUS: Quantity = Dimensionless::of(0.2);
const RHO: Quantity = Dimensionless::of(1.0e2);
const STALL: Quantity = Dimensionless::of(1.0e-6);
const STEP: Quantity = Dimensionless::of(0.01);
const SWEEPS: usize = 20;
const TOLERANCE: Quantity = Dimensionless::of(1.0e-3);

const SOLVER: Settings = Settings {
    backtracks: 27,
    convergence: 1.0e-14,
    history: 10,
    iterations: 50,
};

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
            weights.iter_mut().zip(adjacent).for_each(|(row, node)| {
                row[node] += 1.0;
                row[corner] -= 1.0;
            })
        } else {
            weights
                .iter_mut()
                .zip(&faces)
                .for_each(|(row, (high, low))| {
                    high.iter().for_each(|&node| row[node] += 0.25);
                    low.iter().for_each(|&node| row[node] -= 0.25)
                })
        }
        weights
    })
});

struct Frame {
    edges: CoordinateList<3, 3>,
    jacobian: Quantity<Volume>,
    lengths: TensorList<Quantity<Length>, 3>,
    scaled: Quantity,
}

struct Plane {
    boundary: usize,
    normal: Direction<3>,
    offset: Quantity<Length>,
    slot: usize,
}

struct Energy<'a> {
    boundary: &'a [usize],
    coordinates: Coordinates<3>,
    hexes: &'a [[usize; 8]],
    lambda: TensorListVec<Quantity<Length>, CAP>,
    planes: Vec<Plane>,
    rho: Quantity,
    scales: QuantityVector<Length>,
    theta: Quantity,
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
        let surface = Surface::new(target);
        let mut energy = Energy {
            boundary: &boundary,
            coordinates: coordinates.clone(),
            hexes: &hexes,
            lambda: (0..boundary.len())
                .map(|_| [Quantity::default(); CAP].into())
                .collect(),
            planes: Vec::new(),
            rho: RHO,
            scales: QuantityVector::zero(0),
            theta: Quantity::default(),
        };
        let mut best: Option<Coordinates<3>> = None;
        let mut first = true;
        'ratchet: loop {
            for _ in 0..SWEEPS {
                energy.refresh(&surface);
                let quality = energy.quality();
                let gap = energy.gap();
                if first {
                    first = false;
                    if gap <= TOLERANCE {
                        best = Some(energy.coordinates.clone());
                        energy.theta = Self::next(quality);
                    }
                }
                if quality >= energy.theta && gap <= TOLERANCE {
                    best = Some(energy.coordinates.clone());
                    energy.theta = Self::next(quality);
                    energy.rho *= 2.0;
                    continue 'ratchet;
                }
                let typical = FIRST * energy.typical();
                let mut x = energy.flat();
                let outcome = minimize(&mut energy, &mut x, typical.value(), &SOLVER);
                energy.assign(&x);
                energy.settle();
                let initial = Quantity::<Area>::new(outcome.initial);
                let decrease = (initial - Quantity::<Area>::new(outcome.value))
                    / initial.abs().max(Quantity::new(Scalar::MIN_POSITIVE));
                if decrease < STALL {
                    break;
                }
            }
            break;
        }
        *coordinates = best.ok_or("could not fit the surface")?;
        Ok(())
    }
    fn next(quality: Quantity) -> Quantity {
        ((quality / STEP).floor() + Dimensionless::of(1.0)) * STEP
    }
}

impl Frame {
    fn new(weights: &Weights, nodes: &CoordinateList<3, 8>) -> Self {
        let edges: CoordinateList<3, 3> = weights
            .iter()
            .map(|weight_i| {
                nodes
                    .iter()
                    .zip(weight_i)
                    .filter(|(_, weight)| **weight != 0.0)
                    .fold(Coordinate::default(), |sum, (node, weight)| {
                        sum + node * *weight
                    })
            })
            .collect();
        let jacobian = &edges[0] * edges[1].cross(&edges[2]);
        let lengths: TensorList<Quantity<Length>, 3> =
            edges.iter().map(|edge| edge.norm()).collect();
        let product = lengths[0] * lengths[1] * lengths[2];
        let scaled = if lengths.iter().all(|&length| length > Quantity::default()) {
            jacobian / product
        } else {
            Quantity::default()
        };
        Self {
            edges,
            jacobian,
            lengths,
            scaled,
        }
    }
}

impl Plane {
    fn residual(&self, position: &Coordinate<3>) -> Quantity<Length> {
        position * &self.normal - self.offset
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
        sizes: &QuantityVector<Length>,
        coordinates: &Coordinates<3>,
    ) -> Vec<Plane> {
        let bvh = self.target.bvh();
        let mut planes = Vec::new();
        boundary.iter().enumerate().for_each(|(index, &node)| {
            let position = &coordinates[node];
            let Some((point, first)) = bvh.closest_point(position, self.surface, &self.elements)
            else {
                return;
            };
            let reach = sizes[node] * RADIUS;
            let (low, high) = (
                from_fn(|i| point[i] - reach).into(),
                from_fn(|i| point[i] + reach).into(),
            );
            let query: BoundingBox<3> = [&low, &high]
                .into_iter()
                .collect::<CoordinatesRef<'_, 3>>()
                .into();
            let mut kept = Vec::<Direction<3>>::new();
            for triangle in once(first).chain(bvh.overlapping(&query)) {
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
                let normal = self.normals[triangle].clone();
                if kept.iter().any(|other| (other * &normal).abs() > CREASE) {
                    continue;
                }
                let plane = Plane {
                    boundary: index,
                    offset: &self.surface[self.elements[triangle][0]] * &normal,
                    normal,
                    slot: kept.len(),
                };
                if triangle != first && plane.residual(position).abs() > sizes[node] * TOLERANCE {
                    continue;
                }
                kept.push(plane.normal.clone());
                planes.push(plane)
            }
        });
        planes
    }
}

impl Energy<'_> {
    fn assign(&mut self, x: &[Scalar]) {
        self.coordinates
            .iter_mut()
            .zip(x.as_chunks::<3>().0)
            .for_each(|(node, values)| {
                node.iter_mut()
                    .zip(values)
                    .for_each(|(entry, &value)| *entry = Quantity::new(value))
            })
    }
    fn flat(&self) -> Vec<Scalar> {
        self.coordinates
            .iter()
            .flat_map(|node| node.iter().map(|entry| entry.value()))
            .collect()
    }
    fn refresh(&mut self, surface: &Surface) {
        self.scales = scales(self.hexes, &self.coordinates);
        let sizes = node_scales(
            self.boundary,
            self.hexes,
            &self.scales,
            self.coordinates.len(),
        );
        self.planes = surface.planes(self.boundary, &sizes, &self.coordinates);
    }
    fn typical(&self) -> Quantity<Length> {
        self.scales.iter().copied().sum::<Quantity<Length>>() / self.scales.len().max(1) as Scalar
    }
    fn quality(&self) -> Quantity {
        self.hexes
            .iter()
            .map(|hex| {
                let nodes = hex
                    .iter()
                    .map(|&node| self.coordinates[node].clone())
                    .collect();
                POINTS
                    .iter()
                    .map(|weights| Frame::new(weights, &nodes).scaled)
                    .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
            })
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
    }
    fn gap(&self) -> Quantity {
        let sizes = node_scales(
            self.boundary,
            self.hexes,
            &self.scales,
            self.coordinates.len(),
        );
        self.planes
            .iter()
            .map(|plane| {
                let node = self.boundary[plane.boundary];
                plane.residual(&self.coordinates[node]).abs() / sizes[node]
            })
            .fold(Quantity::default(), Quantity::max)
    }
    fn settle(&mut self) {
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            self.lambda[plane.boundary][plane.slot] +=
                self.rho * plane.residual(&self.coordinates[node])
        })
    }
    fn evaluate(&self, mut gradient: Option<&mut Gradient>) -> Quantity<Area> {
        let mut total = Quantity::<Area>::default();
        self.hexes
            .iter()
            .zip(&self.scales)
            .for_each(|(hex, &scale)| {
                let nodes = hex
                    .iter()
                    .map(|&node| self.coordinates[node].clone())
                    .collect();
                let (value, slopes) = terms(self.theta, scale, &nodes, gradient.is_some());
                total -= value;
                if let Some(gradient) = gradient.as_deref_mut() {
                    hex.iter()
                        .zip(slopes)
                        .for_each(|(&node, slope)| gradient[node] -= slope)
                }
            });
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            let residual = plane.residual(&self.coordinates[node]);
            let multiplier = self.lambda[plane.boundary][plane.slot];
            total += multiplier * residual + self.rho * residual * residual * 0.5;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[node] += &plane.normal * (multiplier + self.rho * residual)
            }
        });
        total
    }
}

impl Objective for Energy<'_> {
    fn value(&mut self, x: &[Scalar]) -> Scalar {
        self.assign(x);
        self.evaluate(None).value()
    }
    fn gradient(&mut self, x: &[Scalar], gradient: &mut [Scalar]) {
        self.assign(x);
        let mut slopes = Gradient::zero(self.coordinates.len());
        self.evaluate(Some(&mut slopes));
        gradient
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(slopes.iter())
            .for_each(|(values, slope)| {
                values
                    .iter_mut()
                    .zip(slope.iter())
                    .for_each(|(value, entry)| *value = entry.value())
            })
    }
}

fn terms(
    theta: Quantity,
    scale: Quantity<Length>,
    nodes: &CoordinateList<3, 8>,
    derivative: bool,
) -> (Quantity<Area>, CoordinateList<3, 8>) {
    let frames: [Frame; 9] = from_fn(|point| Frame::new(&POINTS[point], nodes));
    let (j, lowest_j) = frames
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.jacobian.total_cmp(&b.1.jacobian))
        .map(|(point, frame)| (point, frame.jacobian))
        .unwrap();
    let (s, lowest_sj) = frames
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.scaled.total_cmp(&b.1.scaled))
        .map(|(point, frame)| (point, frame.scaled))
        .unwrap();
    let mut slopes = CoordinateList::default();
    if lowest_j <= Quantity::default() {
        if derivative {
            let e = &frames[j].edges;
            let g = [
                (e[1].cross(&e[2]) / scale).with_unit(),
                (e[2].cross(&e[0]) / scale).with_unit(),
                (e[0].cross(&e[1]) / scale).with_unit(),
            ]
            .into();
            scatter(&POINTS[j], &g, &mut slopes)
        }
        (lowest_j / scale, slopes)
    } else if lowest_sj <= theta {
        let area = scale * scale;
        if derivative {
            let frame = &frames[s];
            let u: DirectionList<3, 3> = frame
                .edges
                .iter()
                .map(|edge| edge.clone().normalized())
                .collect();
            let g = (0..3)
                .map(|i| {
                    let c = u[(i + 1) % 3].cross(&u[(i + 2) % 3]);
                    let along = &c * &u[i];
                    let slope: TensorRank1<3, Reference, ReciprocalLength> =
                        ((c - &u[i] * along) / frame.lengths[i]).with_unit();
                    (slope * area).with_unit()
                })
                .collect();
            scatter(&POINTS[s], &g, &mut slopes)
        }
        (lowest_sj * area, slopes)
    } else {
        (theta * scale * scale, slopes)
    }
}

fn scatter(weights: &Weights, g: &CoordinateList<3, 3>, slopes: &mut CoordinateList<3, 8>) {
    weights.iter().zip(g).for_each(|(weight_i, g_i)| {
        slopes
            .iter_mut()
            .zip(weight_i)
            .for_each(|(slope, &weight)| *slope += g_i * weight)
    })
}

fn scales(hexes: &[[usize; 8]], coordinates: &Coordinates<3>) -> QuantityVector<Length> {
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
                .map(|&[a, b]| (&coordinates[hex[a]] - &coordinates[hex[b]]).norm())
                .sum::<Quantity<Length>>()
                / 12.0
        })
        .collect()
}

fn node_scales(
    boundary: &[usize],
    hexes: &[[usize; 8]],
    scales: &QuantityVector<Length>,
    nodes: usize,
) -> QuantityVector<Length> {
    let mut sum = QuantityVector::<Length>::zero(nodes);
    let mut count = vec![0_usize; nodes];
    hexes.iter().zip(scales).for_each(|(hex, &scale)| {
        hex.iter().for_each(|&node| {
            sum[node] += scale;
            count[node] += 1
        })
    });
    let mut sizes = QuantityVector::<Length>::zero(nodes);
    boundary
        .iter()
        .for_each(|&node| sizes[node] = sum[node] / count[node].max(1) as Scalar);
    sizes
}
