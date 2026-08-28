#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{chi, hexahedron::CORNERS, regularized},
        },
    },
    math::{
        ContractWith, CrossProduct, Quantity, Reference, Scalar, Tensor, TensorRank1,
        TensorRank1List, TensorRank1Vec,
    },
    units::{Area, Dimensionless, Length, ReciprocalLength},
};
use std::{
    array::from_fn,
    collections::VecDeque,
    mem::replace,
    thread::{available_parallelism, scope},
};

type EdgeList = TensorRank1List<3, Reference, 3>;
type Slope = TensorRank1<3, Reference, ReciprocalLength>;
type Gradient = TensorRank1Vec<3, Reference, Dimensionless>;
type Target = (Coordinate<3>, Direction<3>, Quantity<Area>);

const ARMIJO: Scalar = 1.0e-4;
const BACKTRACKS: usize = 32;
const BALANCE: Scalar = 2.5e3;
const CONVERGENCE: Scalar = 1.0e-5;
const CURVATURE_FLOOR: Scalar = 1.0e-12;
const EPSILON_FLOOR: Scalar = 1.0e-12;
const HISTORY: usize = 8;
const ITERATIONS: usize = 100;
const RELAXATION: Scalar = 0.1;
const STAGNATION: Scalar = 5.0e-4;
const SWEEPS: usize = 50;
const TOLERANCE: Scalar = 1.0e-3;
const WEIGHT_FLOOR: Quantity = Dimensionless::of(0.3);
const WINDOW: usize = 3;

/// Queried at boundary-quad centroids to drive the fit energy toward the target
/// geometry. Must be cheap and thread-safe: every sweep projects every quad.
pub(crate) trait Oracle: Sync {
    /// The closest point on the target surface to `query`, and the outward unit
    /// normal there.
    fn project(&self, query: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)>;
}

/// [`Oracle`] backed by a triangulated [`Tessellation`]: BVH closest-point plus
/// the hit triangle's face normal.
pub(super) struct Facets<'a> {
    bvh: &'a BoundingVolumeHierarchy<3>,
    coordinates: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    normals: DirectionsRef<'a, 3>,
}

struct Sweep<'a> {
    epsilon: Scalar,
    hex_chunk: usize,
    hexes: &'a [[usize; 8]],
    lengths: Vec<Quantity<Length>>,
    node_chunk: usize,
    node_quads: &'a [Vec<usize>],
    nodes: &'a [usize],
    scales: Vec<Quantity<Length>>,
    slot: &'a [Option<usize>],
    targets: Vec<Target>,
    tracked: &'a [usize],
    unknowns: usize,
}

impl Mesh<3> {
    pub(super) fn fit<O: Oracle>(
        &mut self,
        nodes: &[usize],
        oracle: &O,
    ) -> Result<(), &'static str> {
        let number_of_nodes = self.number_of_nodes();
        let mut free = vec![false; number_of_nodes];
        nodes.iter().for_each(|&node| free[node] = true);
        let mut hexes = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(block) => hexes.extend(block.iter().copied()),
                _ => return Err("fit requires a hexahedral mesh"),
            }
        }
        let node_hexes = self.node_element_connectivity().to_vec();
        let tracked: Vec<usize> = {
            let mut seen = vec![false; hexes.len()];
            nodes
                .iter()
                .flat_map(|&node| node_hexes[node].iter().copied())
                .filter(|&hex| !replace(&mut seen[hex], true))
                .collect()
        };
        let quads: Vec<[usize; 4]> = self
            .exterior_faces()
            .iter()
            .filter(|face| face.iter().any(|&node| free[node]))
            .map(|face| from_fn(|i| face[i]))
            .collect();
        let mut node_quads = vec![Vec::new(); number_of_nodes];
        quads.iter().enumerate().for_each(|(index, quad)| {
            quad.iter().for_each(|&node| {
                if free[node] {
                    node_quads[node].push(index)
                }
            })
        });
        let neighbors = self.node_node_connectivity().to_vec();
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let quad_chunk = quads.len().div_ceil(threads).max(1);
        let hex_chunk = tracked.len().div_ceil(threads).max(1);
        let node_chunk = nodes.len().div_ceil(threads).max(1);
        let coordinates = self.coordinates.members_mut();
        let mut slot = vec![None; number_of_nodes];
        nodes
            .iter()
            .enumerate()
            .for_each(|(index, &node)| slot[node] = Some(index));
        let unknowns = nodes.len();
        let mut epsilon: Scalar = 1.0;
        let mut previous = Quantity::<Length>::new(Scalar::INFINITY);
        let mut window = VecDeque::<Quantity<Length>>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let (lengths, scales) = sizes(&neighbors, &hexes, coordinates);
            let mut state = Sweep {
                epsilon,
                hex_chunk,
                hexes: &hexes,
                lengths,
                node_chunk,
                node_quads: &node_quads,
                nodes,
                scales,
                slot: &slot,
                targets: project(oracle, &quads, coordinates, quad_chunk)?,
                tracked: &tracked,
                unknowns,
            };
            let (quality, worst) = state.measure(coordinates);
            if sweep > 0 {
                epsilon = schedule(epsilon, quality, previous, worst);
                state.epsilon = epsilon;
            }
            previous = quality;
            let (shift, value, settled) = state.minimize(coordinates);
            let stagnant = window.len() == WINDOW
                && window.iter().fold(value, |high, &entry| high.max(entry))
                    - window.iter().fold(value, |low, &entry| low.min(entry))
                    <= value.abs() * STAGNATION;
            if settled || shift < TOLERANCE || stagnant {
                break;
            }
            if window.len() == WINDOW {
                window.pop_front();
            }
            window.push_back(value);
        }
        Ok(())
    }
}

impl<'a> Facets<'a> {
    pub(super) fn new(target: &'a Tessellation) -> Self {
        let surface = target.mesh();
        Self {
            bvh: target.bvh(),
            coordinates: surface.coordinates(),
            elements: surface.connectivities().iter().flatten().collect(),
            normals: target.normals().iter().flatten().collect(),
        }
    }
}

impl Oracle for Facets<'_> {
    fn project(&self, query: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        self.bvh
            .closest_point(query, self.coordinates, &self.elements)
            .map(|(point, index)| (point, self.normals[index].clone()))
    }
}

/// Projects every boundary-quad centroid onto the target, pairing each hit with
/// the worst tangent-plane deviation among the quad's four nodes.
fn project<O: Oracle>(
    oracle: &O,
    quads: &[[usize; 4]],
    coordinates: &Coordinates<3>,
    chunk: usize,
) -> Result<Vec<Target>, &'static str> {
    let mut targets = vec![None; quads.len()];
    scope(|scope| {
        targets
            .chunks_mut(chunk)
            .zip(quads.chunks(chunk))
            .for_each(|(targets, quads)| {
                scope.spawn(move || {
                    targets.iter_mut().zip(quads).for_each(|(target, quad)| {
                        let centroid = quad
                            .iter()
                            .map(|&node| &coordinates[node])
                            .sum::<Coordinate<3>>()
                            / 4.0;
                        *target = oracle.project(&centroid).map(|(point, normal)| {
                            let distance = quad
                                .iter()
                                .map(|&node| {
                                    let deviation = (&coordinates[node] - &point) * &normal;
                                    deviation * deviation
                                })
                                .fold(Quantity::default(), Quantity::max);
                            (point, normal, distance)
                        });
                    })
                });
            });
    });
    targets
        .into_iter()
        .collect::<Option<_>>()
        .ok_or("no projection onto target surface")
}

impl Sweep<'_> {
    fn measure(&self, coordinates: &Coordinates<3>) -> (Quantity<Length>, Scalar) {
        self.tracked
            .iter()
            .map(|&hex| {
                let scale = self.scales[hex].value();
                (
                    self.scales[hex]
                        * energy(&self.hexes[hex], coordinates, scale.powi(3) * self.epsilon),
                    determinant(&self.hexes[hex], coordinates) / scale.powi(3),
                )
            })
            .fold(
                (Quantity::default(), Scalar::INFINITY),
                |(quality, worst), (q, d)| (quality + q, worst.min(d)),
            )
    }
    fn objective(&self, coordinates: &Coordinates<3>) -> Quantity<Length> {
        scope(|scope| {
            self.tracked
                .chunks(self.hex_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&hex| {
                                self.scales[hex]
                                    * energy(
                                        &self.hexes[hex],
                                        coordinates,
                                        self.scales[hex].value().powi(3) * self.epsilon,
                                    )
                            })
                            .sum::<Quantity<Length>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .sum::<Quantity<Length>>()
        }) + scope(|scope| {
            self.nodes
                .chunks(self.node_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&node| {
                                BALANCE / self.lengths[node]
                                    * self.node_quads[node]
                                        .iter()
                                        .map(|&quad| {
                                            let (point, normal, distance) = &self.targets[quad];
                                            let weight = weight(*distance, self.lengths[node]);
                                            let deviation = (&coordinates[node] - point) * normal;
                                            deviation * deviation * weight
                                        })
                                        .sum::<Quantity<Area>>()
                            })
                            .sum::<Quantity<Length>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .sum::<Quantity<Length>>()
        })
    }
    fn derivative(&self, coordinates: &Coordinates<3>) -> Gradient {
        let mut gradient = scope(|scope| {
            self.tracked
                .chunks(self.hex_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut partial = self.empty();
                        chunk.iter().for_each(|&hex| {
                            let local = scatter(
                                &self.hexes[hex],
                                coordinates,
                                self.scales[hex].value().powi(3) * self.epsilon,
                            );
                            self.hexes[hex]
                                .iter()
                                .zip(local)
                                .for_each(|(&node, contribution)| {
                                    if let Some(index) = self.slot[node] {
                                        partial[index] += contribution * self.scales[hex]
                                    }
                                })
                        });
                        partial
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .fold(self.empty(), |mut sum, partial| {
                    sum += &partial;
                    sum
                })
        });
        scope(|scope| {
            gradient
                .as_mut_slice()
                .chunks_mut(self.node_chunk)
                .zip(self.nodes.chunks(self.node_chunk))
                .for_each(|(entries, nodes)| {
                    scope.spawn(move || {
                        entries.iter_mut().zip(nodes).for_each(|(entry, &node)| {
                            self.node_quads[node].iter().for_each(|&quad| {
                                let (point, normal, distance) = &self.targets[quad];
                                let weight = weight(*distance, self.lengths[node]);
                                let deviation = (&coordinates[node] - point) * normal;
                                let factor =
                                    2.0 * BALANCE / self.lengths[node] * weight * deviation;
                                *entry += normal * factor.value()
                            })
                        });
                    });
                });
        });
        gradient
    }
    fn empty(&self) -> Gradient {
        (0..self.unknowns)
            .map(|_| TensorRank1::const_from([0.0; 3]))
            .collect()
    }
    fn minimize(&self, coordinates: &mut Coordinates<3>) -> (Scalar, Quantity<Length>, bool) {
        let typical = self
            .nodes
            .iter()
            .map(|&node| self.lengths[node])
            .sum::<Quantity<Length>>()
            / self.nodes.len().max(1) as Scalar;
        let mut x: Coordinates<3> = self
            .nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect();
        let anchor = x.clone();
        let mut history = Vec::<(Coordinates<3>, Gradient)>::new();
        let mut gradient = self.derivative(coordinates);
        let mut value = self.objective(coordinates);
        let mut settled = false;
        for iteration in 0..ITERATIONS {
            let magnitude = gradient.norm().value();
            if magnitude / x.norm().value().max(1.0) < CONVERGENCE {
                settled = iteration == 0;
                break;
            }
            let d = direction(&gradient, &history, typical / magnitude);
            let slope = gradient.contract_with(&d);
            if slope >= Quantity::default() {
                history.clear();
                continue;
            }
            let mut step = 1.0;
            let mut accepted = None;
            for _ in 0..BACKTRACKS {
                self.nodes
                    .iter()
                    .enumerate()
                    .for_each(|(index, &node)| coordinates[node] = &x[index] + &d[index] * step);
                let trial = self.objective(coordinates);
                if trial <= value + ARMIJO * step * slope {
                    accepted = Some(trial);
                    break;
                }
                step *= 0.5;
            }
            let Some(trial) = accepted else {
                self.nodes
                    .iter()
                    .enumerate()
                    .for_each(|(index, &node)| coordinates[node] = x[index].clone());
                if history.is_empty() {
                    break;
                }
                history.clear();
                continue;
            };
            let s: Coordinates<3> = d.iter().map(|entry| entry * step).collect();
            x += &s;
            let updated = self.derivative(coordinates);
            let y: Gradient = updated
                .iter()
                .zip(gradient.iter())
                .map(|(new, old)| new - old)
                .collect();
            if s.contract_with(&y) > s.norm() * (CURVATURE_FLOOR * y.norm().value()) {
                if history.len() == HISTORY {
                    history.remove(0);
                }
                history.push((s, y));
            }
            gradient = updated;
            value = trial;
        }
        let shift = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, &node)| {
                ((&x[index] - &anchor[index]).norm() / self.lengths[node]).value()
            })
            .fold(0.0, Scalar::max);
        (shift, value, settled)
    }
}

fn sizes(
    neighbors: &[Vec<usize>],
    hexes: &[[usize; 8]],
    coordinates: &Coordinates<3>,
) -> (Vec<Quantity<Length>>, Vec<Quantity<Length>>) {
    let lengths: Vec<Quantity<Length>> = (0..coordinates.len())
        .map(|node| {
            neighbors[node]
                .iter()
                .map(|&neighbor| (&coordinates[neighbor] - &coordinates[node]).norm())
                .sum::<Quantity<Length>>()
                / neighbors[node].len().max(1) as Scalar
        })
        .collect();
    let scales = hexes
        .iter()
        .map(|hex| {
            hex.iter()
                .map(|&node| lengths[node])
                .sum::<Quantity<Length>>()
                / 8.0
        })
        .collect();
    (lengths, scales)
}

fn schedule(
    epsilon: Scalar,
    quality: Quantity<Length>,
    previous: Quantity<Length>,
    worst: Scalar,
) -> Scalar {
    let sigma = RELAXATION.max((1.0 - quality / previous).value());
    let mu = (1.0 - sigma) * chi(epsilon, worst);
    let epsilon_2021 = if worst < mu {
        2.0 * (mu * (mu - worst)).sqrt()
    } else {
        EPSILON_FLOOR
    };
    let epsilon_1999 = (1.0e-18 + (0.2 * worst).powi(2)).sqrt();
    epsilon_2021.min(epsilon_1999)
}

fn direction(
    gradient: &Gradient,
    history: &[(Coordinates<3>, Gradient)],
    fallback: Quantity<Length>,
) -> Coordinates<3> {
    let mut q = gradient.clone();
    let mut alphas = vec![0.0; history.len()];
    let mut rhos = vec![Quantity::<ReciprocalLength>::default(); history.len()];
    history.iter().enumerate().rev().for_each(|(k, (s, y))| {
        rhos[k] = 1.0 / y.contract_with(s);
        alphas[k] = (s.contract_with(&q) * rhos[k]).value();
        q.iter_mut()
            .zip(y.iter())
            .for_each(|(qi, yi)| *qi -= yi * alphas[k]);
    });
    let mut q: Coordinates<3> = q * history
        .last()
        .map_or(fallback, |(s, y)| s.contract_with(y) / y.norm_squared());
    history.iter().enumerate().for_each(|(k, (s, y))| {
        let beta = (y.contract_with(&q) * rhos[k]).value();
        q.iter_mut()
            .zip(s.iter())
            .for_each(|(qi, si)| *qi += si * (alphas[k] - beta));
    });
    q *= -1.0;
    q
}

fn edges(
    corner: usize,
    adjacent: &[usize; 3],
    hex: &[usize; 8],
    coordinates: &Coordinates<3>,
) -> EdgeList {
    let origin = &coordinates[hex[corner]];
    (0..3)
        .map(|i| (&coordinates[hex[adjacent[i]]] - origin).with_unit())
        .collect()
}

fn weight(distance: Quantity<Area>, length: Quantity<Length>) -> Quantity<Dimensionless> {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

fn energy(hex: &[usize; 8], coordinates: &Coordinates<3>, epsilon: Scalar) -> Scalar {
    CORNERS
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| regularized(&edges(corner, adjacent, hex, coordinates), epsilon))
        .sum()
}

fn scatter(hex: &[usize; 8], coordinates: &Coordinates<3>, epsilon: Scalar) -> [Slope; 8] {
    let mut local = from_fn(|_| TensorRank1::<3, Reference>::const_from([0.0; 3]));
    CORNERS.iter().enumerate().for_each(|(corner, adjacent)| {
        let edges = edges(corner, adjacent, hex, coordinates);
        let trace = edges.norm_squared().value();
        let determinant = edges.scalar_triple_product();
        let denominator = chi(epsilon, determinant);
        let alpha = 3.0 * trace.sqrt() / denominator;
        let beta = trace.powf(1.5)
            * 0.5
            * (1.0 + determinant / (epsilon * epsilon + determinant * determinant).sqrt())
            / (denominator * denominator);
        let crosses = [
            edges[1].cross(&edges[2]),
            edges[2].cross(&edges[0]),
            edges[0].cross(&edges[1]),
        ];
        (0..3).for_each(|i| {
            local[corner] += &crosses[i] * beta - &edges[i] * alpha;
            local[adjacent[i]] += &edges[i] * alpha - &crosses[i] * beta;
        });
    });
    local.map(|entry| entry.with_unit())
}

fn determinant(hex: &[usize; 8], coordinates: &Coordinates<3>) -> Scalar {
    CORNERS
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| edges(corner, adjacent, hex, coordinates).scalar_triple_product())
        .fold(Scalar::INFINITY, Scalar::min)
}
