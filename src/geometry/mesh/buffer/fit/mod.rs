#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{chi, hexahedron, pyramid, regularized, tetrahedron, wedge},
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

type CornerTable = [(usize, [usize; 3])];

const ARMIJO: Scalar = 1.0e-4;
const BACKTRACKS: usize = 32;
const BALANCE: Scalar = 2.5e3;
const CONVERGENCE: Scalar = 1.0e-5;
/// A boundary node within this many local edge lengths of a crease curve,
/// *before any fit sweep moves it*, is treated as sitting on it. A pre-fit
/// octree-dual boundary node is typically a bit over one local length from
/// the true B-rep surface (it is a blocky Cartesian approximation, not yet
/// fit to anything) -- calibrated on `capped_cylinder`'s rim by sweeping this
/// constant and measuring the fitted rim's radius/height error: 0.5-1.05
/// caught too few nodes to matter, 2.0 caught nodes far enough from the
/// crease that including them made the fit *worse*. 1.35 was the best of the
/// values tried on that one fixture; not yet validated against a real
/// crease-tangle case (see `cad/REVIEW.md`).
const CREASE_TOLERANCE: Scalar = 1.35;
/// A crease-owned boundary node within this many local edge lengths of a hard
/// corner point ([`Brep::features`](crate::geometry::cad::brep::Brep::features)
/// corners) pins toward it (a weighted energy term, not a hard constraint --
/// still tightens by 1-2 orders of magnitude in practice) instead of sliding
/// freely along its crease curve's tangent. Several creases converge at a
/// corner, so "along the curve" is not one direction there -- the same
/// instability the crease term itself exists to remove, one level up.
/// Uncalibrated against a real multi-junction fixture (see `cad/REVIEW.md`);
/// set equal to `CREASE_TOLERANCE` since a pre-fit node this close to its own
/// crease curve's endpoint is exactly the node a corner term should catch.
const CORNER_TOLERANCE: Scalar = CREASE_TOLERANCE;
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
    /// A discrete id for whatever surface region `query`'s nearest point
    /// belongs to, if this oracle can distinguish regions at all. `None` --
    /// the default -- disables the topological crease-ownership gate in
    /// [`Mesh::fit`] rather than denying every node.
    fn feature(&self, _query: &Coordinate<3>) -> Option<usize> {
        None
    }
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
    crease_targets: &'a [Option<CreaseTarget>],
    element_chunk: usize,
    elements: &'a [(&'static CornerTable, Vec<usize>)],
    epsilon: Scalar,
    lengths: Vec<Quantity<Length>>,
    node_chunk: usize,
    node_faces: &'a [Vec<usize>],
    nodes: &'a [usize],
    scales: Vec<Quantity<Length>>,
    slot: &'a [Option<usize>],
    targets: Vec<Target>,
    tracked: &'a [usize],
    unknowns: usize,
}

/// A crease-owned node's fit target: the nearest point on its crease curve set
/// at the start of this sweep, the curve's local unit tangent there (all-zero
/// if degenerate), and the squared perpendicular deviation, for [`weight`].
type CreaseTarget = (Coordinate<3>, [Scalar; 3], Quantity<Area>);

impl Mesh<3> {
    pub(super) fn fit<O: Oracle>(
        &mut self,
        nodes: &[usize],
        oracle: &O,
        creases: &[(Vec<Coordinate<3>>, Vec<usize>)],
        corner_points: &[Coordinate<3>],
    ) -> Result<(), &'static str> {
        let mut elements: Vec<(&'static CornerTable, Vec<usize>)> = Vec::new();
        for block in self.iter() {
            let corners: &'static CornerTable = match block {
                Connectivity::Hexahedral(_) => &hexahedron::CORNERS,
                Connectivity::Tetrahedral(_) => &tetrahedron::CORNERS,
                Connectivity::Pyramidal(_) => &pyramid::CORNERS,
                Connectivity::Wedge(_) => &wedge::CORNERS,
                _ => return Err("fit requires hexahedra, tetrahedra, pyramids or wedges"),
            };
            elements.extend(block.iter().map(|element| (corners, element.to_vec())));
        }
        let number_of_nodes = self.number_of_nodes();
        let mut free = vec![false; number_of_nodes];
        nodes.iter().for_each(|&node| free[node] = true);
        let node_elements = self.node_element_connectivity().to_vec();
        let tracked: Vec<usize> = {
            let mut seen = vec![false; elements.len()];
            nodes
                .iter()
                .flat_map(|&node| node_elements[node].iter().copied())
                .filter(|&element| !replace(&mut seen[element], true))
                .collect()
        };
        let faces: Vec<Vec<usize>> = self
            .exterior_faces()
            .into_iter()
            .filter(|face| face.iter().any(|&node| free[node]))
            .collect();
        let mut node_faces = vec![Vec::new(); number_of_nodes];
        faces.iter().enumerate().for_each(|(index, face)| {
            face.iter().for_each(|&node| {
                if free[node] {
                    node_faces[node].push(index)
                }
            })
        });
        let neighbors = self.node_node_connectivity().to_vec();
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let face_chunk = faces.len().div_ceil(threads).max(1);
        let element_chunk = tracked.len().div_ceil(threads).max(1);
        let node_chunk = nodes.len().div_ceil(threads).max(1);
        let coordinates = self.coordinates.members_mut();
        let mut slot = vec![None; number_of_nodes];
        nodes
            .iter()
            .enumerate()
            .for_each(|(index, &node)| slot[node] = Some(index));
        let unknowns = nodes.len();
        let curve_only: Vec<Vec<Coordinate<3>>> =
            creases.iter().map(|(curve, _)| curve.clone()).collect();
        // Which boundary nodes a crease curve owns, and *which curve*, decided
        // once from the pre-fit mesh -- the same "freeze early" principle that
        // keeps a face target from a nearest-face flip, applied at the node
        // level instead. Freezing the curve identity too (not just the
        // ownership bool) matters whenever two creases pass close together
        // (a thin flange's top and bottom rim): re-searching all curves fresh
        // every sweep would let a node between them flip which one it targets
        // as it moves, reintroducing the same discrete-flip instability this
        // mechanism exists to remove, one level down.
        //
        // Proximity to the curve alone is also not enough: a node can sit
        // just as close to an unrelated crease's curve as to its own (the far
        // side of a thin flange, say). `touches_one_of` requires that at
        // least one of the node's own incident faces currently projects (via
        // `oracle.feature`) onto one of *this* curve's bordering faces --
        // topological ownership, not just Euclidean distance. An oracle that
        // cannot report features (`feature` returns `None` everywhere) skips
        // this gate entirely, preserving old behaviour.
        let (crease_curve, corner_owned): (Vec<Option<usize>>, Vec<Option<usize>>) =
            if creases.is_empty() {
                (vec![None; number_of_nodes], vec![None; number_of_nodes])
            } else {
                let (initial_lengths, _) = sizes(&neighbors, &elements, coordinates);
                let crease_curve: Vec<Option<usize>> = (0..number_of_nodes)
                    .map(|node| -> Option<usize> {
                        if node_faces[node].is_empty() {
                            return None;
                        }
                        let (index, _, distance, _) =
                            nearest_on_polylines(&curve_only, &coordinates[node])?;
                        if distance > CREASE_TOLERANCE * initial_lengths[node].value() {
                            return None;
                        }
                        touches_one_of(
                            oracle,
                            &faces,
                            &node_faces[node],
                            coordinates,
                            &creases[index].1,
                        )
                        .then_some(index)
                    })
                    .collect();
                // Several creases converge at a hard corner, so a crease-owned
                // node that lands here has no single tangent to slide along --
                // pin it to the exact corner point instead. Frozen alongside
                // crease_curve for the same reason: recomputing which corner is
                // nearest every sweep from a moving position could flip between
                // two close corners, reintroducing the instability this whole
                // mechanism exists to remove. Only a node the crease term already
                // owns is a candidate -- an unrelated node merely passing near a
                // corner (a different part of a complex model) must not pin here.
                let corner_owned: Vec<Option<usize>> = (0..number_of_nodes)
                    .map(|node| {
                        crease_curve[node]?;
                        nearest_point(corner_points, &coordinates[node]).and_then(
                            |(index, distance)| {
                                (distance <= CORNER_TOLERANCE * initial_lengths[node].value())
                                    .then_some(index)
                            },
                        )
                    })
                    .collect();
                (crease_curve, corner_owned)
            };
        let mut epsilon: Scalar = 1.0;
        let mut previous = Quantity::<Length>::new(Scalar::INFINITY);
        let mut window = VecDeque::<Quantity<Length>>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let (lengths, scales) = sizes(&neighbors, &elements, coordinates);
            let crease_targets: Vec<Option<CreaseTarget>> = (0..number_of_nodes)
                .map(|node| {
                    if let Some(index) = corner_owned[node] {
                        let point = corner_points[index].clone();
                        let distance = (&coordinates[node] - &point).norm().value();
                        return Some((point, [0.0; 3], Quantity::<Area>::new(distance * distance)));
                    }
                    crease_curve[node].map(|index| {
                        let (_, point, distance, tangent) = nearest_on_polylines(
                            std::slice::from_ref(&curve_only[index]),
                            &coordinates[node],
                        )
                        .expect("a crease-owned node has a nearest point on its frozen curve");
                        (point, tangent, Quantity::<Area>::new(distance * distance))
                    })
                })
                .collect();
            let mut state = Sweep {
                crease_targets: &crease_targets,
                element_chunk,
                elements: &elements,
                epsilon,
                lengths,
                node_chunk,
                node_faces: &node_faces,
                nodes,
                scales,
                slot: &slot,
                targets: project(oracle, &faces, coordinates, face_chunk)?,
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
    faces: &[Vec<usize>],
    coordinates: &Coordinates<3>,
    chunk: usize,
) -> Result<Vec<Target>, &'static str> {
    let mut targets = vec![None; faces.len()];
    scope(|scope| {
        targets
            .chunks_mut(chunk)
            .zip(faces.chunks(chunk))
            .for_each(|(targets, faces)| {
                scope.spawn(move || {
                    targets.iter_mut().zip(faces).for_each(|(target, face)| {
                        let centroid = face
                            .iter()
                            .map(|&node| &coordinates[node])
                            .sum::<Coordinate<3>>()
                            / face.len() as Scalar;
                        *target = oracle.project(&centroid).map(|(point, normal)| {
                            let distance = face
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

/// The closest point over the union of `curves` to `query`, which curve
/// (index into `curves`) it landed on, its distance, and the unit tangent of
/// the segment it landed on (all-zero if that segment is degenerate) --
/// `None` only when `curves` is empty. Plain polylines, not a `cad`-specific
/// type: this stays generic over whatever supplied them. A single-curve slice
/// restricts the search to that curve, for tracking a node against a curve
/// already chosen (see `crease_curve` in [`Mesh::fit`]).
fn nearest_on_polylines(
    curves: &[Vec<Coordinate<3>>],
    query: &Coordinate<3>,
) -> Option<(usize, Coordinate<3>, Scalar, [Scalar; 3])> {
    let point: [Scalar; 3] = from_fn(|k| query[k].value());
    let mut best: Option<(usize, Coordinate<3>, Scalar, [Scalar; 3])> = None;
    for (curve_index, curve) in curves.iter().enumerate() {
        for pair in curve.windows(2) {
            let a: [Scalar; 3] = from_fn(|k| pair[0][k].value());
            let b: [Scalar; 3] = from_fn(|k| pair[1][k].value());
            let edge: [Scalar; 3] = from_fn(|k| b[k] - a[k]);
            let span = edge.iter().map(|x| x * x).sum::<Scalar>();
            let t = if span > 0.0 {
                ((0..3).map(|k| (point[k] - a[k]) * edge[k]).sum::<Scalar>() / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let foot: [Scalar; 3] = from_fn(|k| a[k] + t * edge[k]);
            let distance = (0..3)
                .map(|k| (point[k] - foot[k]).powi(2))
                .sum::<Scalar>()
                .sqrt();
            if best
                .as_ref()
                .is_none_or(|&(_, _, best_distance, _)| distance < best_distance)
            {
                let tangent = if span > 0.0 {
                    let norm = span.sqrt();
                    from_fn(|k| edge[k] / norm)
                } else {
                    [0.0; 3]
                };
                best = Some((curve_index, Coordinate::from(foot), distance, tangent));
            }
        }
    }
    best
}

/// The index into `points` nearest `query`, and its distance -- `None` only
/// when `points` is empty.
fn nearest_point(points: &[Coordinate<3>], query: &Coordinate<3>) -> Option<(usize, Scalar)> {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| (index, (point - query).norm().value()))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}

/// Whether any of `node`'s incident boundary faces currently projects (via
/// `oracle.feature` at the face's centroid) onto one of `faces`. If the
/// oracle never reports a feature id (every face gives `None`), the gate does
/// not apply -- every node passes, matching the old (topology-blind)
/// behaviour for an oracle with no concept of discrete surface regions.
fn touches_one_of<O: Oracle>(
    oracle: &O,
    boundary_faces: &[Vec<usize>],
    node_faces: &[usize],
    coordinates: &Coordinates<3>,
    faces: &[usize],
) -> bool {
    let mut saw_a_feature = false;
    for &face in node_faces {
        let boundary_face = &boundary_faces[face];
        let centroid = boundary_face
            .iter()
            .map(|&node| &coordinates[node])
            .sum::<Coordinate<3>>()
            / boundary_face.len() as Scalar;
        if let Some(id) = oracle.feature(&centroid) {
            saw_a_feature = true;
            if faces.contains(&id) {
                return true;
            }
        }
    }
    !saw_a_feature
}

impl Sweep<'_> {
    fn measure(&self, coordinates: &Coordinates<3>) -> (Quantity<Length>, Scalar) {
        self.tracked
            .iter()
            .map(|&element| {
                let scale = self.scales[element].value();
                let (corners, nodes) = &self.elements[element];
                (
                    self.scales[element]
                        * energy(corners, nodes, coordinates, scale.powi(3) * self.epsilon),
                    determinant(corners, nodes, coordinates) / scale.powi(3),
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
                .chunks(self.element_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&element| {
                                let (corners, nodes) = &self.elements[element];
                                self.scales[element]
                                    * energy(
                                        corners,
                                        nodes,
                                        coordinates,
                                        self.scales[element].value().powi(3) * self.epsilon,
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
                                    * match &self.crease_targets[node] {
                                        Some((point, tangent, distance)) => {
                                            let weight = weight(*distance, self.lengths[node]);
                                            crease_term(&coordinates[node], point, tangent).0
                                                * weight
                                        }
                                        None => self.node_faces[node]
                                            .iter()
                                            .map(|&face| {
                                                let (point, normal, distance) = &self.targets[face];
                                                let weight = weight(*distance, self.lengths[node]);
                                                let deviation =
                                                    (&coordinates[node] - point) * normal;
                                                deviation * deviation * weight
                                            })
                                            .sum::<Quantity<Area>>(),
                                    }
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
                .chunks(self.element_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut partial = self.empty();
                        chunk.iter().for_each(|&element| {
                            let (corners, nodes) = &self.elements[element];
                            let local = scatter(
                                corners,
                                nodes,
                                coordinates,
                                self.scales[element].value().powi(3) * self.epsilon,
                            );
                            nodes.iter().zip(local).for_each(|(&node, contribution)| {
                                if let Some(index) = self.slot[node] {
                                    partial[index] += contribution * self.scales[element]
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
                            match &self.crease_targets[node] {
                                Some((point, tangent, distance)) => {
                                    let weight = weight(*distance, self.lengths[node]);
                                    let perp = crease_term(&coordinates[node], point, tangent).1;
                                    let factor =
                                        (2.0 * BALANCE / self.lengths[node] * weight).value();
                                    *entry += TensorRank1::const_from(perp) * factor
                                }
                                None => self.node_faces[node].iter().for_each(|&face| {
                                    let (point, normal, distance) = &self.targets[face];
                                    let weight = weight(*distance, self.lengths[node]);
                                    let deviation = (&coordinates[node] - point) * normal;
                                    let factor =
                                        2.0 * BALANCE / self.lengths[node] * weight * deviation;
                                    *entry += normal * factor.value()
                                }),
                            }
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
    elements: &[(&'static CornerTable, Vec<usize>)],
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
    let scales = elements
        .iter()
        .map(|(_, element)| {
            element
                .iter()
                .map(|&node| lengths[node])
                .sum::<Quantity<Length>>()
                / element.len() as Scalar
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
    element: &[usize],
    coordinates: &Coordinates<3>,
) -> EdgeList {
    let origin = &coordinates[element[corner]];
    (0..3)
        .map(|i| (&coordinates[element[adjacent[i]]] - origin).with_unit())
        .collect()
}

/// The crease-line squared deviation `|delta|² − (delta·t)²` (`delta = x −
/// point`) and its gradient `2·(delta − (delta·t)·t)` w.r.t. `x`, both raw
/// (unit-stripped, matching how `nearest_on_polylines` already works). This
/// pulls a node onto the curve but leaves it free to slide along `tangent`;
/// an all-zero `tangent` (a degenerate segment) falls back to plain point
/// attraction, since `delta·t = 0` then and `delta` passes through unchanged.
fn crease_term(
    x: &Coordinate<3>,
    point: &Coordinate<3>,
    tangent: &[Scalar; 3],
) -> (Quantity<Area>, [Scalar; 3]) {
    let delta: [Scalar; 3] = from_fn(|k| x[k].value() - point[k].value());
    let along = (0..3).map(|k| delta[k] * tangent[k]).sum::<Scalar>();
    let perp: [Scalar; 3] = from_fn(|k| delta[k] - along * tangent[k]);
    let squared = perp.iter().map(|p| p * p).sum::<Scalar>();
    (Quantity::<Area>::new(squared.max(0.0)), perp)
}

fn weight(distance: Quantity<Area>, length: Quantity<Length>) -> Quantity<Dimensionless> {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

fn energy(
    corners: &CornerTable,
    element: &[usize],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> Scalar {
    corners
        .iter()
        .map(|(corner, adjacent)| {
            regularized(&edges(*corner, adjacent, element, coordinates), epsilon)
        })
        .sum()
}

fn scatter(
    corners: &CornerTable,
    element: &[usize],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> [Slope; 8] {
    let mut local: [TensorRank1<3, Reference>; 8] = from_fn(|_| TensorRank1::const_from([0.0; 3]));
    corners.iter().for_each(|(corner, adjacent)| {
        let edges = edges(*corner, adjacent, element, coordinates);
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
            local[*corner] += &crosses[i] * beta - &edges[i] * alpha;
            local[adjacent[i]] += &edges[i] * alpha - &crosses[i] * beta;
        });
    });
    local.map(|entry| entry.with_unit())
}

fn determinant(corners: &CornerTable, element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    corners
        .iter()
        .map(|(corner, adjacent)| {
            edges(*corner, adjacent, element, coordinates).scalar_triple_product()
        })
        .fold(Scalar::INFINITY, Scalar::min)
}
