#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates, CoordinatesRef,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{chi, hexahedron::CORNERS, regularized},
        },
    },
    math::{CrossProduct, Scalar, Tensor},
};
use std::{
    array::from_fn,
    collections::VecDeque,
    mem::replace,
    thread::{available_parallelism, scope},
};

const ARMIJO: Scalar = 1.0e-4;
const BACKTRACKS: usize = 32;
const BALANCE: Scalar = 2.5e3;
const CONVERGENCE: Scalar = 1.0e-5;
const CURVATURE_FLOOR: Scalar = 1.0e-12;
const EPSILON_FLOOR: Scalar = 1.0e-12;
const HISTORY: usize = 8;
const ITERATIONS: usize = 100;
const RELAXATION: Scalar = 0.1;
const STAGNATION: Scalar = 1.0e-4;
const SWEEPS: usize = 100;
const TOLERANCE: Scalar = 1.0e-3;
const WEIGHT_FLOOR: Scalar = 0.3;
const WINDOW: usize = 5;

impl Mesh<3> {
    pub(super) fn fit(
        &mut self,
        nodes: &[usize],
        target: &Tessellation,
    ) -> Result<(), &'static str> {
        let surface = target.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: CoordinatesRef<3> = target.normals().iter().flatten().collect();
        let bvh = target.bvh();
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
        let chunk_size = quads.len().div_ceil(threads).max(1);
        let node_chunk = nodes.len().div_ceil(threads).max(1);
        let coordinates = self.coordinates.members_mut();
        let mut slot = vec![None; number_of_nodes];
        nodes
            .iter()
            .enumerate()
            .for_each(|(index, &node)| slot[node] = Some(index));
        let unknowns = 3 * nodes.len();
        let mut epsilon: Scalar = 1.0;
        let mut previous = Scalar::INFINITY;
        let mut window = VecDeque::<Scalar>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let lengths: Vec<Scalar> = (0..number_of_nodes)
                .map(|node| {
                    neighbors[node]
                        .iter()
                        .map(|&neighbor| (&coordinates[neighbor] - &coordinates[node]).norm())
                        .sum::<Scalar>()
                        / neighbors[node].len().max(1) as Scalar
                })
                .collect();
            let scales: Vec<Scalar> = hexes
                .iter()
                .map(|hex| hex.iter().map(|&node| lengths[node]).sum::<Scalar>() / 8.0)
                .collect();
            let mut targets = vec![None; quads.len()];
            scope(|scope| {
                let coordinates = &*coordinates;
                let (elements, normals) = (&elements, &normals);
                targets
                    .chunks_mut(chunk_size)
                    .zip(quads.chunks(chunk_size))
                    .for_each(|(targets, quads)| {
                        scope.spawn(move || {
                            targets.iter_mut().zip(quads).for_each(|(target, quad)| {
                                let centroid = quad
                                    .iter()
                                    .map(|&node| &coordinates[node])
                                    .sum::<Coordinate<3>>()
                                    / 4.0;
                                *target = bvh
                                    .closest_point(&centroid, surface_coordinates, elements)
                                    .map(|(point, index)| {
                                        let normal = normals[index].clone();
                                        let distance = quad
                                            .iter()
                                            .map(|&node| {
                                                let deviation =
                                                    (&coordinates[node] - &point) * &normal;
                                                deviation * deviation
                                            })
                                            .fold(0.0, Scalar::max);
                                        (point, normal, distance)
                                    });
                            })
                        });
                    });
            });
            let targets: Vec<(Coordinate<3>, Coordinate<3>, Scalar)> = targets
                .into_iter()
                .collect::<Option<_>>()
                .ok_or("empty tessellation")?;
            let (quality, worst) = tracked
                .iter()
                .map(|&hex| {
                    let scale = scales[hex];
                    (
                        scale * energy(&hexes[hex], coordinates, scale.powi(3) * epsilon),
                        determinant(&hexes[hex], coordinates) / scale.powi(3),
                    )
                })
                .fold((0.0, Scalar::INFINITY), |(quality, worst), (q, d)| {
                    (quality + q, worst.min(d))
                });
            if sweep > 0 {
                let sigma = RELAXATION.max(1.0 - quality / previous);
                let mu = (1.0 - sigma) * chi(epsilon, worst);
                let epsilon_2021 = if worst < mu {
                    2.0 * (mu * (mu - worst)).sqrt()
                } else {
                    EPSILON_FLOOR
                };
                let epsilon_1999 = (1.0e-18 + (0.2 * worst).powi(2)).sqrt();
                epsilon = epsilon_2021.min(epsilon_1999);
            }
            previous = quality;
            let hex_chunk = tracked.len().div_ceil(threads).max(1);
            let (hexes_ref, scales_ref) = (&hexes, &scales);
            let (node_quads_ref, lengths_ref) = (&node_quads, &lengths);
            let objective = |coordinates: &Coordinates<3>| -> Scalar {
                scope(|scope| {
                    tracked
                        .chunks(hex_chunk)
                        .map(|chunk| {
                            scope.spawn(move || {
                                chunk
                                    .iter()
                                    .map(|&hex| {
                                        scales_ref[hex]
                                            * energy(
                                                &hexes_ref[hex],
                                                coordinates,
                                                scales_ref[hex].powi(3) * epsilon,
                                            )
                                    })
                                    .sum::<Scalar>()
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .sum::<Scalar>()
                }) + scope(|scope| {
                    let targets_ref = &targets;
                    nodes
                        .chunks(node_chunk)
                        .map(|chunk| {
                            scope.spawn(move || {
                                chunk
                                    .iter()
                                    .map(|&node| {
                                        BALANCE / lengths_ref[node]
                                            * node_quads_ref[node]
                                                .iter()
                                                .map(|&quad| {
                                                    let (point, normal, distance) =
                                                        &targets_ref[quad];
                                                    let weight = 1.0
                                                        / (distance
                                                            / (lengths_ref[node]
                                                                * lengths_ref[node]))
                                                            .max(WEIGHT_FLOOR);
                                                    let deviation =
                                                        (&coordinates[node] - point) * normal;
                                                    weight * deviation * deviation
                                                })
                                                .sum::<Scalar>()
                                    })
                                    .sum::<Scalar>()
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .sum::<Scalar>()
                })
            };
            let slot_ref = &slot;
            let derivative = |coordinates: &Coordinates<3>| -> Vec<Scalar> {
                let mut flat = scope(|scope| {
                    tracked
                        .chunks(hex_chunk)
                        .map(|chunk| {
                            scope.spawn(move || {
                                let mut partial = vec![0.0; unknowns];
                                chunk.iter().for_each(|&hex| {
                                    let local = scatter(
                                        &hexes_ref[hex],
                                        coordinates,
                                        scales_ref[hex].powi(3) * epsilon,
                                    );
                                    hexes_ref[hex].iter().zip(local).for_each(
                                        |(&node, contribution)| {
                                            if let Some(index) = slot_ref[node] {
                                                (0..3).for_each(|ax| {
                                                    partial[3 * index + ax] +=
                                                        contribution[ax] * scales_ref[hex]
                                                });
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
                        .fold(vec![0.0; unknowns], |mut sum, partial| {
                            sum.iter_mut()
                                .zip(&partial)
                                .for_each(|(entry, term)| *entry += term);
                            sum
                        })
                });
                scope(|scope| {
                    let targets_ref = &targets;
                    flat.chunks_mut(3 * node_chunk)
                        .zip(nodes.chunks(node_chunk))
                        .for_each(|(flat_chunk, node_chunk_slice)| {
                            scope.spawn(move || {
                                flat_chunk.chunks_mut(3).zip(node_chunk_slice).for_each(
                                    |(entry, &node)| {
                                        node_quads_ref[node].iter().for_each(|&quad| {
                                            let (point, normal, distance) = &targets_ref[quad];
                                            let weight = 1.0
                                                / (distance
                                                    / (lengths_ref[node] * lengths_ref[node]))
                                                    .max(WEIGHT_FLOOR);
                                            let deviation = (&coordinates[node] - point) * normal;
                                            let factor = 2.0 * BALANCE / lengths_ref[node]
                                                * weight
                                                * deviation;
                                            (0..3).for_each(|ax| entry[ax] += normal[ax] * factor);
                                        })
                                    },
                                );
                            });
                        });
                });
                flat
            };
            let typical = nodes.iter().map(|&node| lengths[node]).sum::<Scalar>()
                / nodes.len().max(1) as Scalar;
            let mut x: Vec<Scalar> = nodes
                .iter()
                .flat_map(|&node| (0..3).map(|ax| coordinates[node][ax]).collect::<Vec<_>>())
                .collect();
            let anchor = x.clone();
            let mut history = Vec::<(Vec<Scalar>, Vec<Scalar>)>::new();
            let mut flat = derivative(coordinates);
            let mut value = objective(coordinates);
            let mut settled = false;
            for iteration in 0..ITERATIONS {
                let magnitude = norm(&flat);
                if magnitude / norm(&x).max(1.0) < CONVERGENCE {
                    settled = iteration == 0;
                    break;
                }
                let d = direction(&flat, &history, typical / magnitude);
                let slope = dot(&flat, &d);
                if slope >= 0.0 {
                    history.clear();
                    continue;
                }
                let mut step = 1.0;
                let mut accepted = None;
                for _ in 0..BACKTRACKS {
                    nodes.iter().enumerate().for_each(|(index, &node)| {
                        (0..3).for_each(|ax| {
                            coordinates[node][ax] = x[3 * index + ax] + step * d[3 * index + ax]
                        })
                    });
                    let trial = objective(coordinates);
                    if trial <= value + ARMIJO * step * slope {
                        accepted = Some(trial);
                        break;
                    }
                    step *= 0.5;
                }
                let Some(trial) = accepted else {
                    nodes.iter().enumerate().for_each(|(index, &node)| {
                        (0..3).for_each(|ax| coordinates[node][ax] = x[3 * index + ax])
                    });
                    if history.is_empty() {
                        break;
                    }
                    history.clear();
                    continue;
                };
                let s: Vec<Scalar> = d.iter().map(|entry| entry * step).collect();
                x.iter_mut().zip(&s).for_each(|(entry, si)| *entry += si);
                let updated = derivative(coordinates);
                let y: Vec<Scalar> = updated
                    .iter()
                    .zip(&flat)
                    .map(|(new, old)| new - old)
                    .collect();
                if dot(&s, &y) > CURVATURE_FLOOR * norm(&s) * norm(&y) {
                    if history.len() == HISTORY {
                        history.remove(0);
                    }
                    history.push((s, y));
                }
                flat = updated;
                value = trial;
            }
            let shift = nodes
                .iter()
                .enumerate()
                .map(|(index, &node)| {
                    (0..3)
                        .map(|ax| (x[3 * index + ax] - anchor[3 * index + ax]).powi(2))
                        .sum::<Scalar>()
                        .sqrt()
                        / lengths[node]
                })
                .fold(0.0, Scalar::max);
            let stagnant =
                window.len() == WINDOW && (window[0] - value).abs() <= STAGNATION * value.abs();
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

fn dot(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn norm(a: &[Scalar]) -> Scalar {
    dot(a, a).sqrt()
}

fn direction(
    gradient: &[Scalar],
    history: &[(Vec<Scalar>, Vec<Scalar>)],
    fallback: Scalar,
) -> Vec<Scalar> {
    let mut q = gradient.to_vec();
    let mut alphas = vec![0.0; history.len()];
    let mut rhos = vec![0.0; history.len()];
    history.iter().enumerate().rev().for_each(|(k, (s, y))| {
        rhos[k] = 1.0 / dot(y, s);
        alphas[k] = rhos[k] * dot(s, &q);
        q.iter_mut()
            .zip(y)
            .for_each(|(qi, yi)| *qi -= alphas[k] * yi);
    });
    let gamma = history
        .last()
        .map_or(fallback, |(s, y)| dot(s, y) / dot(y, y));
    q.iter_mut().for_each(|qi| *qi *= gamma);
    history.iter().enumerate().for_each(|(k, (s, y))| {
        let beta = rhos[k] * dot(y, &q);
        q.iter_mut()
            .zip(s)
            .for_each(|(qi, si)| *qi += (alphas[k] - beta) * si);
    });
    q.iter_mut().for_each(|qi| *qi = -*qi);
    q
}

fn edges(
    corner: usize,
    adjacent: &[usize; 3],
    hex: &[usize; 8],
    coordinates: &Coordinates<3>,
) -> CoordinateList<3, 3> {
    let origin = &coordinates[hex[corner]];
    (0..3)
        .map(|i| &coordinates[hex[adjacent[i]]] - origin)
        .collect()
}

fn energy(hex: &[usize; 8], coordinates: &Coordinates<3>, epsilon: Scalar) -> Scalar {
    CORNERS
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| regularized(&edges(corner, adjacent, hex, coordinates), epsilon))
        .sum()
}

fn scatter(hex: &[usize; 8], coordinates: &Coordinates<3>, epsilon: Scalar) -> [Coordinate<3>; 8] {
    let mut local = from_fn(|_| Coordinate::const_from([0.0; 3]));
    CORNERS.iter().enumerate().for_each(|(corner, adjacent)| {
        let edges = edges(corner, adjacent, hex, coordinates);
        let trace = edges.norm_squared();
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
    local
}

fn determinant(hex: &[usize; 8], coordinates: &Coordinates<3>) -> Scalar {
    CORNERS
        .iter()
        .enumerate()
        .map(|(corner, adjacent)| edges(corner, adjacent, hex, coordinates).scalar_triple_product())
        .fold(Scalar::INFINITY, Scalar::min)
}
