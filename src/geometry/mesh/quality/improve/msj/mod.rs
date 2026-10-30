#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates, CoordinatesRef, Direction, DirectionList,
        DirectionsRef,
        bbox::BoundingBox,
        mesh::{Connectivity, Mesh, MeshError, Tessellation},
    },
    math::{
        CrossProduct, Quantity, QuantityVector, Reference, Scalar, Tensor, TensorList,
        TensorListVec, TensorRank1, TensorRank1Vec,
        optimize::{Armijo, Corrections, LimitedMemoryBFGS, Tolerances},
    },
    units::{Area, Dimensionless, Length, ReciprocalLength, Volume},
};
use std::{
    array::from_fn,
    iter::once,
    mem::swap,
    sync::{LazyLock, Mutex},
    thread::{available_parallelism, scope},
};

type Gradient = TensorRank1Vec<3, Reference, Length>;
type Weights = [[Scalar; 8]; 3];

const CAP: usize = 4;
const LIGHT: usize = 1 << 15;
const SERIAL: usize = 512;
const CREASE: Quantity = Dimensionless::of(0.866);
const FIRST: Quantity = Dimensionless::of(0.01);
const RADIUS: Quantity = Dimensionless::of(0.2);
const MARGIN: Scalar = 1.0e-9;
const SAFETY: Scalar = 1.0e-6;
const RHO: Quantity = Dimensionless::of(1.0e2);
const STALL: Quantity = Dimensionless::of(1.0e-6);
const STEP: Quantity = Dimensionless::of(0.01);
const TOLERANCE: Quantity = Dimensionless::of(1.0e-3);

fn solver(typical: Quantity<Length>) -> LimitedMemoryBFGS<Armijo, Length> {
    LimitedMemoryBFGS {
        abs_tol: Tolerances {
            constraint: 0.0,
            residual: 1.0e-14,
        },
        history: 10,
        line_search: Armijo {
            control: 1.0e-4,
            cut_back: 0.5,
            max_steps: 27,
        },
        max_steps: 50,
        rel_tol: None,
        typical,
    }
}

struct Point {
    sparse: [Vec<(usize, Scalar)>; 3],
    weights: Weights,
}

static POINTS: LazyLock<[Point; 9]> = LazyLock::new(|| {
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
        let sparse = from_fn(|i| {
            (0..8)
                .filter(|&node| weights[i][node] != 0.0)
                .map(|node| (node, weights[i][node]))
                .collect()
        });
        Point { sparse, weights }
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

type Bound = (Quantity, Quantity<Length>);
type Part = (Quantity<Area>, CoordinateList<3, 8>, Bound);

#[derive(Default)]
struct Scratch {
    base: Option<(Coordinates<3>, Quantity)>,
    bounds: Vec<Bound>,
    fresh: Vec<Bound>,
    moved: Vec<Quantity<Area>>,
    parts: Vec<Part>,
    values: Vec<Quantity<Area>>,
}

struct Reach<'a> {
    bounds: &'a [Bound],
    moved: &'a [Quantity<Area>],
}

impl Reach<'_> {
    fn within(&self, index: usize, hex: &[usize; 8]) -> Option<Quantity<Length>> {
        let (margin, shortest) = self.bounds[index];
        let reach = (margin - Dimensionless::of(MARGIN)) * shortest / 12.0 * (1.0 - SAFETY);
        if reach <= Quantity::default() {
            return None;
        }
        let limit = reach * reach;
        let mut farthest = Quantity::default();
        for &node in hex {
            let squared = self.moved[node];
            if squared >= limit {
                return None;
            }
            farthest = farthest.max(squared)
        }
        Some(farthest.sqrt())
    }
    fn decayed(&self, index: usize, moved: Quantity<Length>) -> Bound {
        let (margin, shortest) = self.bounds[index];
        let margin = margin - moved / shortest * 12.0;
        if margin > Dimensionless::of(MARGIN) {
            (margin, shortest - moved * 2.0)
        } else {
            Default::default()
        }
    }
}

struct Energy<'a> {
    boundary: &'a [usize],
    hexes: &'a [[usize; 8]],
    lambda: TensorListVec<Quantity<Length>, CAP>,
    planes: Vec<Plane>,
    rho: Quantity,
    scales: QuantityVector<Length>,
    scratch: Mutex<Scratch>,
    theta: Quantity,
}

impl Mesh<3> {
    pub fn improve_scaled_jacobian(
        &mut self,
        target: &Tessellation,
        threshold: Quantity,
        sweeps: usize,
    ) -> Result<(), MeshError> {
        let mut hexes = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(elements) => hexes.extend(elements.iter().copied()),
                _ => return Err(MeshError::HexahedraRequired),
            }
        }
        let mut boundary: Vec<usize> = self.exterior_faces().into_iter().flatten().collect();
        boundary.sort_unstable();
        boundary.dedup();
        let coordinates = &mut self.coordinates;
        let surface = Surface::new(target);
        let mut energy = Energy {
            boundary: &boundary,
            hexes: &hexes,
            lambda: (0..boundary.len())
                .map(|_| [Quantity::default(); CAP].into())
                .collect(),
            planes: Vec::new(),
            rho: RHO,
            scales: QuantityVector::zero(0),
            scratch: Mutex::default(),
            theta: Quantity::default(),
        };
        let mut current = coordinates.clone();
        let mut best: Option<Coordinates<3>> = None;
        let mut first = true;
        let mut corrections = Corrections::default();
        let stop = 'ratchet: loop {
            let mut stalled = false;
            let mut sweep = 0;
            loop {
                energy.refresh(&surface, &current);
                let quality = energy.quality(&current);
                let gap = energy.gap(&current);
                if first {
                    first = false;
                    if gap <= TOLERANCE {
                        best = Some(current.clone());
                        if quality >= threshold {
                            break 'ratchet None;
                        }
                        energy.theta = Self::next(quality);
                    }
                }
                if quality >= energy.theta && gap <= TOLERANCE {
                    best = Some(current.clone());
                    if quality >= threshold {
                        break 'ratchet None;
                    }
                    energy.theta = Self::next(quality);
                    continue 'ratchet;
                }
                if stalled {
                    break 'ratchet Some(MeshError::Stall {
                        theta: energy.theta.value(),
                        sweeps: sweep,
                    });
                }
                if sweep == sweeps {
                    break 'ratchet Some(MeshError::Budget {
                        theta: energy.theta.value(),
                        sweeps,
                    });
                }
                let descent = solver(FIRST * energy.typical())
                    .descend_from(
                        &mut corrections,
                        |x| Ok(energy.evaluate(x, None)),
                        |x| {
                            let mut slopes = Gradient::zero(x.len());
                            energy.evaluate(x, Some(&mut slopes));
                            Ok(slopes)
                        },
                        current,
                        Vec::new(),
                    )
                    .map_err(|_| MeshError::Minimization)?;
                let decrease = (descent.initial - descent.value)
                    / descent
                        .initial
                        .abs()
                        .max(Quantity::new(Scalar::MIN_POSITIVE));
                current = descent.solution;
                energy.settle(&current);
                sweep += 1;
                if decrease < STALL {
                    stalled = true;
                }
            }
        };
        *coordinates = best.ok_or(MeshError::SurfaceFit)?;
        stop.map_or(Ok(()), Err)
    }
    fn next(quality: Quantity) -> Quantity {
        ((quality / STEP).floor() + Dimensionless::of(1.0)) * STEP
    }
}

impl Frame {
    fn new(point: &Point, nodes: &CoordinateList<3, 8>) -> Self {
        let edges: CoordinateList<3, 3> = point
            .sparse
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .fold(Coordinate::default(), |sum, &(node, weight)| {
                        sum + &nodes[node] * weight
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
    fn refresh(&mut self, surface: &Surface, coordinates: &Coordinates<3>) {
        self.scales = scales(self.hexes, coordinates);
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes = surface.planes(self.boundary, &sizes, coordinates);
    }
    fn typical(&self) -> Quantity<Length> {
        self.scales.iter().copied().sum::<Quantity<Length>>() / self.scales.len().max(1) as Scalar
    }
    fn quality(&self, coordinates: &Coordinates<3>) -> Quantity {
        self.hexes
            .iter()
            .map(|hex| {
                let nodes = hex.iter().map(|&node| coordinates[node].clone()).collect();
                POINTS
                    .iter()
                    .map(|weights| Frame::new(weights, &nodes).scaled)
                    .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
            })
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
    }
    fn gap(&self, coordinates: &Coordinates<3>) -> Quantity {
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes
            .iter()
            .map(|plane| {
                let node = self.boundary[plane.boundary];
                plane.residual(&coordinates[node]).abs() / sizes[node]
            })
            .fold(Quantity::default(), Quantity::max)
    }
    fn settle(&mut self, coordinates: &Coordinates<3>) {
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            self.lambda[plane.boundary][plane.slot] += self.rho * plane.residual(&coordinates[node])
        })
    }
    fn map_hexes<R>(
        &self,
        coordinates: &Coordinates<3>,
        skip: Option<Reach>,
        derivative: bool,
        mapped: &mut [R],
        map: impl Fn(Part) -> R + Sync,
    ) where
        R: Send,
    {
        let chunk = chunk_size(self.hexes.len());
        let map = &map;
        let skip = &skip;
        scope(|scope| {
            mapped
                .chunks_mut(chunk)
                .zip(self.hexes.chunks(chunk))
                .enumerate()
                .for_each(|(index, (mapped, hexes))| {
                    scope.spawn(move || {
                        mapped.iter_mut().zip(hexes).enumerate().for_each(
                            |(offset, (slot, hex))| {
                                let at = index * chunk + offset;
                                *slot = map(
                                    if let Some((skip, moved)) = skip.as_ref().and_then(|skip| {
                                        skip.within(at, hex).map(|moved| (skip, moved))
                                    }) {
                                        let bound = if derivative {
                                            skip.decayed(at, moved)
                                        } else {
                                            Default::default()
                                        };
                                        (Default::default(), Default::default(), bound)
                                    } else {
                                        let nodes = hex
                                            .iter()
                                            .map(|&node| coordinates[node].clone())
                                            .collect();
                                        let scale = self.scales[at];
                                        if derivative {
                                            bounded(self.theta, scale, &nodes, true)
                                        } else {
                                            let (value, slopes) =
                                                terms(self.theta, scale, &nodes, false);
                                            (value, slopes, Default::default())
                                        }
                                    },
                                )
                            },
                        )
                    });
                })
        });
    }
    fn evaluate(
        &self,
        coordinates: &Coordinates<3>,
        mut gradient: Option<&mut Gradient>,
    ) -> Quantity<Area> {
        let mut total = Quantity::<Area>::default();
        let mut scratch = self.scratch.lock().unwrap();
        let Scratch {
            base,
            bounds,
            fresh,
            moved,
            parts,
            values,
        } = &mut *scratch;
        let skip = match base.as_ref().filter(|(_, theta)| *theta == self.theta) {
            Some((base, _)) => {
                displacements(coordinates, base, moved);
                Some(Reach {
                    bounds,
                    moved: &*moved,
                })
            }
            None => None,
        };
        match gradient.as_deref_mut() {
            Some(gradient) => {
                parts.resize_with(self.hexes.len(), Default::default);
                self.map_hexes(coordinates, skip, true, parts, |part| part);
                fresh.clear();
                parts
                    .iter()
                    .zip(self.hexes)
                    .for_each(|((value, slopes, bound), hex)| {
                        total -= *value;
                        fresh.push(*bound);
                        if slopes
                            .iter()
                            .any(|slope| (0..3).any(|i| slope[i] != Quantity::default()))
                        {
                            hex.iter()
                                .zip(slopes.iter())
                                .for_each(|(&node, slope)| gradient[node] -= slope)
                        }
                    });
                swap(bounds, fresh);
                *base = Some((coordinates.clone(), self.theta))
            }
            None => {
                values.resize_with(self.hexes.len(), Default::default);
                self.map_hexes(coordinates, skip, false, values, |(value, _, _)| value);
                values.iter().for_each(|&value| total -= value)
            }
        }
        drop(scratch);
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            let residual = plane.residual(&coordinates[node]);
            let multiplier = self.lambda[plane.boundary][plane.slot];
            total += multiplier * residual + self.rho * residual * residual * 0.5;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[node] += &plane.normal * (multiplier + self.rho * residual)
            }
        });
        total
    }
}

fn displacements(
    coordinates: &Coordinates<3>,
    base: &Coordinates<3>,
    moved: &mut Vec<Quantity<Area>>,
) {
    moved.resize_with(coordinates.len(), Default::default);
    let chunk = if moved.len() < LIGHT {
        moved.len().max(1)
    } else {
        chunk_size(moved.len())
    };
    scope(|scope| {
        moved
            .chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, moved)| {
                scope.spawn(move || {
                    moved.iter_mut().enumerate().for_each(|(offset, moved)| {
                        let at = index * chunk + offset;
                        let step = &coordinates[at] - &base[at];
                        *moved = &step * &step
                    })
                });
            })
    });
}

fn chunk_size(items: usize) -> usize {
    if items < 2 * SERIAL {
        items.max(1)
    } else {
        items
            .div_ceil(available_parallelism().map_or(1, |threads| threads.get()))
            .max(1)
    }
}

fn terms(
    theta: Quantity,
    scale: Quantity<Length>,
    nodes: &CoordinateList<3, 8>,
    derivative: bool,
) -> (Quantity<Area>, CoordinateList<3, 8>) {
    let (value, slopes, _) = bounded(theta, scale, nodes, derivative);
    (value, slopes)
}

fn bounded(
    theta: Quantity,
    scale: Quantity<Length>,
    nodes: &CoordinateList<3, 8>,
    derivative: bool,
) -> Part {
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
            scatter(&POINTS[j].weights, &g, &mut slopes)
        }
        (
            lowest_j / scale - theta * scale * scale,
            slopes,
            Default::default(),
        )
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
            scatter(&POINTS[s].weights, &g, &mut slopes)
        }
        ((lowest_sj - theta) * area, slopes, Default::default())
    } else {
        let margin = lowest_sj - theta.max(Quantity::default());
        let bound = if derivative && margin > Dimensionless::of(MARGIN) {
            let shortest = frames
                .iter()
                .flat_map(|frame| frame.lengths.iter().copied())
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
            (margin, shortest)
        } else {
            Default::default()
        };
        (Quantity::default(), slopes, bound)
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
