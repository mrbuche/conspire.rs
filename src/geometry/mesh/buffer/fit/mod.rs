#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{chi, hexahedron, regularized, tetrahedron},
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
const SMOOTH_CONE: Scalar = 0.94;
const STAGNATION: Scalar = 5.0e-4;
const SWEEPS: usize = 50;
const TOLERANCE: Scalar = 1.0e-3;
const WEIGHT_FLOOR: Quantity = Dimensionless::of(0.3);
const WINDOW: usize = 3;

struct Oracle<'a> {
    bvh: &'a BoundingVolumeHierarchy<3>,
    coordinates: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    normals: DirectionsRef<'a, 3>,
    /// Angle-weighted vertex normals of the target surface. Interpolated at
    /// the closest point they give a normal field continuous across facet
    /// boundaries, so a node sliding tangentially does not see the target
    /// plane jump when its closest facet flips (Protais et al. §4.2.1).
    vertex_normals: Vec<Direction<3>>,
}

/// One pass of the fit over a mesh of `N`-node elements.
struct Sweep<'a, const N: usize> {
    corners: &'static [[usize; 3]; N],
    element_chunk: usize,
    elements: &'a [[usize; N]],
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

impl Mesh<3> {
    pub(crate) fn fit(
        &mut self,
        nodes: &[usize],
        target: &Tessellation,
    ) -> Result<(), &'static str> {
        let mut hexes = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(block) => hexes.extend(block.iter().copied()),
                _ => return Err("fit requires a hexahedral mesh"),
            }
        }
        self.fit_elements::<8, 4>(nodes, target, &hexahedron::CORNERS, &hexes)
    }
    pub(super) fn fit_tets(
        &mut self,
        nodes: &[usize],
        target: &Tessellation,
    ) -> Result<(), &'static str> {
        let mut tets = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Tetrahedral(block) => tets.extend(block.iter().copied()),
                _ => return Err("fit requires a tetrahedral mesh"),
            }
        }
        self.fit_elements::<4, 3>(nodes, target, &tetrahedron::CORNERS, &tets)
    }
    /// Balances element quality against the distance to the target over the
    /// given nodes, for elements of any arity whose corners each meet three
    /// edges.
    fn fit_elements<const N: usize, const F: usize>(
        &mut self,
        nodes: &[usize],
        target: &Tessellation,
        corners: &'static [[usize; 3]; N],
        elements: &[[usize; N]],
    ) -> Result<(), &'static str> {
        let oracle = Oracle::new(target);
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
        let faces: Vec<[usize; F]> = self
            .exterior_faces()
            .iter()
            .filter(|face| face.iter().any(|&node| free[node]))
            .map(|face| from_fn(|i| face[i]))
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
        let mut epsilon: Scalar = 1.0;
        let mut previous = Quantity::<Length>::new(Scalar::INFINITY);
        let mut window = VecDeque::<Quantity<Length>>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let (lengths, scales) = sizes(&neighbors, elements, coordinates);
            let mut state = Sweep {
                corners,
                element_chunk,
                elements,
                epsilon,
                lengths,
                node_chunk,
                node_faces: &node_faces,
                nodes,
                scales,
                slot: &slot,
                targets: oracle.targets(&faces, coordinates, face_chunk)?,
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

impl<'a> Oracle<'a> {
    fn new(target: &'a Tessellation) -> Self {
        let surface = target.mesh();
        Self {
            bvh: target.bvh(),
            coordinates: surface.coordinates(),
            elements: surface.connectivities().iter().flatten().collect(),
            normals: target.normals().iter().flatten().collect(),
            vertex_normals: vertex_normals(surface),
        }
    }
    fn targets<const F: usize>(
        &self,
        faces: &[[usize; F]],
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
                                / F as Scalar;
                            *target = self
                                .bvh
                                .closest_point(&centroid, self.coordinates, &self.elements)
                                .map(|(point, index)| {
                                    let normal = self.smooth_normal(index, &point);
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
            .ok_or("empty tessellation")
    }
    /// The surface normal at `point`, taken to lie on triangle `index`, as the
    /// barycentric blend of that triangle's vertex normals: a field continuous
    /// across facet boundaries wherever the surface is smooth.
    ///
    /// Falls back to the flat facet normal for a degenerate triangle, and
    /// wherever the blend leans more than [`SMOOTH_CONE`] off the facet — the
    /// signature of a crease, where a blended normal would pull nodes off the
    /// feature instead of onto it.
    fn smooth_normal(&self, index: usize, point: &Coordinate<3>) -> Direction<3> {
        let facet = self.normals[index].clone();
        let triangle = self.elements[index];
        let a = &self.coordinates[triangle[0]];
        let b = &self.coordinates[triangle[1]];
        let c = &self.coordinates[triangle[2]];
        let (v0, v1, v2) = (b - a, c - a, point - a);
        let d00 = (&v0 * &v0).value();
        let d01 = (&v0 * &v1).value();
        let d11 = (&v1 * &v1).value();
        let d20 = (&v2 * &v0).value();
        let d21 = (&v2 * &v1).value();
        let denominator = d00 * d11 - d01 * d01;
        if denominator.abs() < CURVATURE_FLOOR {
            return facet;
        }
        let beta = (d11 * d20 - d01 * d21) / denominator;
        let gamma = (d00 * d21 - d01 * d20) / denominator;
        let alpha = 1.0 - beta - gamma;
        let mut normal = &self.vertex_normals[triangle[0]] * alpha;
        normal += &self.vertex_normals[triangle[1]] * beta;
        normal += &self.vertex_normals[triangle[2]] * gamma;
        let normal = normal.normalized();
        if normal.contract_with(&facet).value() < SMOOTH_CONE {
            facet
        } else {
            normal
        }
    }
}

/// Angle-weighted vertex normals of a triangulated surface.
fn vertex_normals(surface: &Mesh<3>) -> Vec<Direction<3>> {
    let coordinates = surface.coordinates();
    let mut normals = vec![TensorRank1::<3, Reference>::const_from([0.0; 3]); coordinates.len()];
    surface
        .connectivities()
        .iter()
        .flatten()
        .for_each(|triangle| {
            let node = [triangle[0], triangle[1], triangle[2]];
            let point = node.map(|node| &coordinates[node]);
            let facet = (point[1] - point[0])
                .cross(point[2] - point[0])
                .normalized();
            (0..3).for_each(|corner| {
                let here = point[corner];
                let one = (point[(corner + 1) % 3] - here).normalized();
                let two = (point[(corner + 2) % 3] - here).normalized();
                let angle = (&one * &two).value().clamp(-1.0, 1.0).acos();
                normals[node[corner]] += &facet * angle;
            })
        });
    normals
        .into_iter()
        .map(|normal| normal.normalized())
        .collect()
}

impl<const N: usize> Sweep<'_, N> {
    fn measure(&self, coordinates: &Coordinates<3>) -> (Quantity<Length>, Scalar) {
        self.tracked
            .iter()
            .map(|&element| {
                let scale = self.scales[element].value();
                (
                    self.scales[element]
                        * energy(
                            self.corners,
                            &self.elements[element],
                            coordinates,
                            scale.powi(3) * self.epsilon,
                        ),
                    determinant(self.corners, &self.elements[element], coordinates) / scale.powi(3),
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
                                self.scales[element]
                                    * energy(
                                        self.corners,
                                        &self.elements[element],
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
                                    * self.node_faces[node]
                                        .iter()
                                        .map(|&face| {
                                            let (point, normal, distance) = &self.targets[face];
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
                .chunks(self.element_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut partial = self.empty();
                        chunk.iter().for_each(|&element| {
                            let local = scatter(
                                self.corners,
                                &self.elements[element],
                                coordinates,
                                self.scales[element].value().powi(3) * self.epsilon,
                            );
                            self.elements[element].iter().zip(local).for_each(
                                |(&node, contribution)| {
                                    if let Some(index) = self.slot[node] {
                                        partial[index] += contribution * self.scales[element]
                                    }
                                },
                            )
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
                            self.node_faces[node].iter().for_each(|&face| {
                                let (point, normal, distance) = &self.targets[face];
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

fn sizes<const N: usize>(
    neighbors: &[Vec<usize>],
    elements: &[[usize; N]],
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
        .map(|element| {
            element
                .iter()
                .map(|&node| lengths[node])
                .sum::<Quantity<Length>>()
                / N as Scalar
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

fn edges<const N: usize>(
    corner: usize,
    adjacent: &[usize; 3],
    element: &[usize; N],
    coordinates: &Coordinates<3>,
) -> EdgeList {
    let origin = &coordinates[element[corner]];
    (0..3)
        .map(|i| (&coordinates[element[adjacent[i]]] - origin).with_unit())
        .collect()
}

fn weight(distance: Quantity<Area>, length: Quantity<Length>) -> Quantity<Dimensionless> {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

fn energy<const N: usize>(
    corners: &[[usize; 3]; N],
    element: &[usize; N],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> Scalar {
    corners
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| {
            regularized(&edges(corner, adjacent, element, coordinates), epsilon)
        })
        .sum()
}

fn scatter<const N: usize>(
    corners: &[[usize; 3]; N],
    element: &[usize; N],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> [Slope; N] {
    let mut local = from_fn(|_| TensorRank1::<3, Reference>::const_from([0.0; 3]));
    corners.iter().enumerate().for_each(|(corner, adjacent)| {
        let edges = edges(corner, adjacent, element, coordinates);
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

fn determinant<const N: usize>(
    corners: &[[usize; 3]; N],
    element: &[usize; N],
    coordinates: &Coordinates<3>,
) -> Scalar {
    corners
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| {
            edges(corner, adjacent, element, coordinates).scalar_triple_product()
        })
        .fold(Scalar::INFINITY, Scalar::min)
}
