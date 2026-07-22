#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            remesh::triangles::{Constraints, Label, remesh},
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{CrossProduct, FxHashMap, Scalar, Tensor},
};
use std::{
    array::from_fn,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet, hash_map::Entry},
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;
const CURVE_SEGMENTS: usize = 16;
const CURVE_RELAXATION: usize = 100;
const CORNER_ANCHOR: Scalar = 0.25;
const SLIVER_RATIO: Scalar = 0.3;
const CONTRACT_RATIO: Scalar = 0.25;
const SNAP_RINGS: usize = 3;
const ROUTE_PENALTY: Scalar = 4.0;
const SHARE_PENALTY: Scalar = 4.0;
const ROUTE_CAP: Scalar = 16.0;
const SURFACE_REFINEMENT: Scalar = 0.5;

#[derive(PartialEq)]
struct Priority(Scalar);

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub struct ProjectedNetwork {
    pub core: Mesh<D>,
    pub surface: Mesh<D>,
    pub quads: Vec<Vec<usize>>,
    pub faces: Vec<Vec<usize>>,
    pub edges: Vec<[usize; 2]>,
    pub curves: Vec<Vec<Coordinate<D>>>,
}

pub struct Imprint {
    pub core: Mesh<D>,
    pub surface: Mesh<D>,
    pub quads: Vec<Vec<usize>>,
    pub faces: Vec<Vec<usize>>,
    pub edges: Vec<[usize; 2]>,
    pub paths: Vec<Vec<usize>>,
    pub patches: Vec<Vec<usize>>,
}

impl Tessellation {
    pub fn fitted_core_and_surface(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<(Mesh<D>, Mesh<D>), &'static str> {
        let mut octree = Octree::<u16, usize>::from_features(self, scale, curvature, 0);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mut core = octree.dualize();
        self.trim(&mut core, self.bvh(), STITCH_TRIM_MARGIN);
        let quads = core.exterior_faces();
        let sizing = QuadSizing::new(&core, &quads)?;
        let elements: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let mut connectivity: Vec<[usize; 3]> = elements
            .iter()
            .map(|element| from_fn(|i| element[i]))
            .collect();
        let mut coordinates = self.mesh().coordinates().clone();
        remesh(
            &mut connectivity,
            &mut coordinates,
            iterations,
            |_, points, _| sizing.target_lengths(points, SURFACE_REFINEMENT),
            None,
        )?;
        let surface = Mesh::from((vec![connectivity.into()], coordinates));
        Ok((core, surface))
    }
    pub fn projected_network(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<ProjectedNetwork, &'static str> {
        let (core, surface) =
            self.fitted_core_and_surface(balancing, scale, curvature, iterations)?;
        let quads = core.exterior_faces();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let surface_coordinates = surface.coordinates();
        let bvh = BoundingVolumeHierarchy::from(&surface);
        let core_coordinates = core.coordinates();
        let mut edges: Vec<[usize; 2]> = quads
            .iter()
            .flat_map(|quad| {
                (0..4).map(|i| {
                    let mut edge = [quad[i], quad[(i + 1) % 4]];
                    edge.sort_unstable();
                    edge
                })
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let project = |point: &Coordinate<D>| -> Result<Coordinate<D>, &'static str> {
            Ok(bvh
                .closest_point(point, surface_coordinates, &elements)
                .ok_or("empty surface")?
                .0)
        };
        let mut corners: HashMap<usize, Coordinate<D>> = HashMap::new();
        edges.iter().flatten().try_for_each(|&vertex| {
            if let Entry::Vacant(slot) = corners.entry(vertex) {
                slot.insert(project(&core_coordinates[vertex])?);
            }
            Ok::<(), &'static str>(())
        })?;
        let mut curves = edges
            .iter()
            .map(|&[a, b]| {
                let mut curve = vec![corners[&a].clone()];
                (1..CURVE_SEGMENTS).try_for_each(|i| {
                    let t = i as Scalar / CURVE_SEGMENTS as Scalar;
                    let sample =
                        &core_coordinates[a] + &((&core_coordinates[b] - &core_coordinates[a]) * t);
                    curve.push(project(&sample)?);
                    Ok::<(), &'static str>(())
                })?;
                curve.push(corners[&b].clone());
                Ok(curve)
            })
            .collect::<Result<Vec<Vec<Coordinate<D>>>, &'static str>>()?;
        let mut incident: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
        edges.iter().enumerate().for_each(|(index, &[a, b])| {
            incident.entry(a).or_default().push((index, true));
            incident.entry(b).or_default().push((index, false));
        });
        let mut vertices: Vec<usize> = incident.keys().copied().collect();
        vertices.sort_unstable();
        (0..CURVE_RELAXATION).try_for_each(|_| {
            curves.iter_mut().try_for_each(|curve| {
                (1..curve.len() - 1).try_for_each(|i| {
                    let midpoint = (&curve[i - 1] + &curve[i + 1]) / 2.0;
                    curve[i] = project(&midpoint)?;
                    Ok::<(), &'static str>(())
                })
            })?;
            vertices.iter().try_for_each(|vertex| {
                let list = &incident[vertex];
                let mean = list
                    .iter()
                    .map(|&(edge, at_start)| {
                        let curve = &curves[edge];
                        if at_start {
                            curve[1].clone()
                        } else {
                            curve[curve.len() - 2].clone()
                        }
                    })
                    .sum::<Coordinate<D>>()
                    / list.len() as Scalar;
                let blend = &(mean * (1.0 - CORNER_ANCHOR)) + &(&corners[vertex] * CORNER_ANCHOR);
                let point = project(&blend)?;
                list.iter().for_each(|&(edge, at_start)| {
                    let curve = &mut curves[edge];
                    if at_start {
                        curve[0] = point.clone();
                    } else {
                        let last = curve.len() - 1;
                        curve[last] = point.clone();
                    }
                });
                Ok::<(), &'static str>(())
            })
        })?;
        let mut edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
        quads.iter().enumerate().for_each(|(index, quad)| {
            (0..4).for_each(|i| {
                let mut edge = [quad[i], quad[(i + 1) % 4]];
                edge.sort_unstable();
                edge_owners.entry(edge).or_default().push(index);
            })
        });
        if edge_owners.values().any(|owners| owners.len() != 2) {
            return Err("non-manifold quad boundary");
        }
        let area = |points: &[Coordinate<D>]| -> Scalar {
            let centroid = points.iter().cloned().sum::<Coordinate<D>>() / points.len() as Scalar;
            (0..points.len())
                .map(|i| {
                    let u = &points[i] - &centroid;
                    let v = &points[(i + 1) % points.len()] - &centroid;
                    u.cross(&v)
                })
                .sum::<Coordinate<D>>()
                .norm()
                / 2.0
        };
        let ratios: Vec<Scalar> = quads
            .iter()
            .map(|quad| {
                let mut boundary = Vec::new();
                (0..4).for_each(|i| {
                    let (a, b) = (quad[i], quad[(i + 1) % 4]);
                    let edge = [a.min(b), a.max(b)];
                    let curve = &curves[edges.binary_search(&edge).unwrap()];
                    let run: Vec<Coordinate<D>> = if a < b {
                        curve.clone()
                    } else {
                        curve.iter().rev().cloned().collect()
                    };
                    run.iter()
                        .take(run.len() - 1)
                        .for_each(|point| boundary.push(point.clone()));
                });
                let footprint: Vec<Coordinate<D>> = quad
                    .iter()
                    .map(|&node| core_coordinates[node].clone())
                    .collect();
                area(&boundary) / area(&footprint)
            })
            .collect();
        let mut root: Vec<usize> = (0..quads.len()).collect();
        fn find(root: &mut [usize], quad: usize) -> usize {
            if root[quad] == quad {
                quad
            } else {
                let top = find(root, root[quad]);
                root[quad] = top;
                top
            }
        }
        let mut order: Vec<usize> = (0..quads.len()).collect();
        order.sort_by(|&x, &y| ratios[x].total_cmp(&ratios[y]));
        order
            .into_iter()
            .filter(|&quad| ratios[quad] < SLIVER_RATIO)
            .for_each(|quad| {
                let donor = (0..4)
                    .filter_map(|i| {
                        let (a, b) = (quads[quad][i], quads[quad][(i + 1) % 4]);
                        let edge = [a.min(b), a.max(b)];
                        edge_owners[&edge]
                            .iter()
                            .copied()
                            .find(|&owner| owner != quad)
                    })
                    .max_by(|&x, &y| ratios[x].total_cmp(&ratios[y]));
                if let Some(donor) = donor {
                    let (a, b) = (find(&mut root, quad), find(&mut root, donor));
                    if a != b {
                        root[a] = b;
                    }
                }
            });
        edges.iter().zip(curves.iter()).for_each(|(edge, curve)| {
            let length = (0..curve.len() - 1)
                .map(|i| (&curve[i + 1] - &curve[i]).norm())
                .sum::<Scalar>();
            let footprint = (&core_coordinates[edge[0]] - &core_coordinates[edge[1]]).norm();
            if length < CONTRACT_RATIO * footprint {
                let owners = &edge_owners[edge];
                let (a, b) = (find(&mut root, owners[0]), find(&mut root, owners[1]));
                if a != b {
                    root[a] = b;
                }
            }
        });
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        (0..quads.len()).for_each(|quad| {
            let top = find(&mut root, quad);
            groups.entry(top).or_default().push(quad);
        });
        let mut faces: Vec<Vec<usize>> = groups.into_values().collect();
        faces.sort_unstable();
        let kept: Vec<usize> = edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                let owners = &edge_owners[*edge];
                find(&mut root, owners[0]) != find(&mut root, owners[1])
            })
            .map(|(index, _)| index)
            .collect();
        let edges: Vec<[usize; 2]> = kept.iter().map(|&index| edges[index]).collect();
        let curves: Vec<Vec<Coordinate<D>>> = kept
            .into_iter()
            .map(|index| std::mem::take(&mut curves[index]))
            .collect();
        Ok(ProjectedNetwork {
            core,
            surface,
            quads,
            faces,
            edges,
            curves,
        })
    }
    pub fn imprinted_network(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<Imprint, &'static str> {
        imprint(
            self.projected_network(balancing, scale, curvature, iterations)?,
            iterations,
        )
    }
}

fn polyline_distance(point: &Coordinate<D>, polyline: &[Coordinate<D>]) -> (Scalar, Coordinate<D>) {
    (0..polyline.len() - 1)
        .map(|i| {
            let ab = &polyline[i + 1] - &polyline[i];
            let ap = point - &polyline[i];
            let t = ((&ap * &ab) / (&ab * &ab)).clamp(0.0, 1.0);
            let closest = &polyline[i] + &(ab * t);
            ((&closest - point).norm(), closest)
        })
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .unwrap()
}

enum Retry {
    Fatal(&'static str),
    Merge(Vec<usize>),
}

fn find(root: &mut [usize], quad: usize) -> usize {
    if root[quad] == quad {
        quad
    } else {
        let top = find(root, root[quad]);
        root[quad] = top;
        top
    }
}

fn merge_network(
    faces: &mut Vec<Vec<usize>>,
    edges: &mut Vec<[usize; 2]>,
    curves: &mut Vec<Vec<Coordinate<D>>>,
    edge_owners: &HashMap<[usize; 2], Vec<usize>>,
    quads: &[Vec<usize>],
    failed: &[usize],
) {
    let mut root: Vec<usize> = (0..quads.len()).collect();
    faces
        .iter()
        .for_each(|group| group.iter().for_each(|&quad| root[quad] = group[0]));
    failed.iter().for_each(|&index| {
        let owners = &edge_owners[&edges[index]];
        let (a, b) = (find(&mut root, owners[0]), find(&mut root, owners[1]));
        if a != b {
            root[a] = b;
        }
    });
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    (0..quads.len()).for_each(|quad| {
        let top = find(&mut root, quad);
        groups.entry(top).or_default().push(quad);
    });
    *faces = groups.into_values().collect();
    faces.sort_unstable();
    let kept: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| {
            let owners = &edge_owners[*edge];
            find(&mut root, owners[0]) != find(&mut root, owners[1])
        })
        .map(|(index, _)| index)
        .collect();
    *edges = kept.iter().map(|&index| edges[index]).collect();
    *curves = kept
        .into_iter()
        .map(|index| std::mem::take(&mut curves[index]))
        .collect();
}

fn imprint(network: ProjectedNetwork, iterations: usize) -> Result<Imprint, &'static str> {
    let ProjectedNetwork {
        core,
        surface,
        quads,
        mut faces,
        mut edges,
        mut curves,
    } = network;
    let triangles: Vec<[usize; 3]> = surface
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| from_fn(|i| triangle[i]))
        .collect();
    let base = surface.coordinates().clone();
    let number_of_vertices = base.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); number_of_vertices];
    triangles.iter().for_each(|triangle| {
        (0..3).for_each(|i| {
            adjacency[triangle[i]].push(triangle[(i + 1) % 3]);
            adjacency[triangle[i]].push(triangle[(i + 2) % 3]);
        })
    });
    adjacency.iter_mut().for_each(|neighbors| {
        neighbors.sort_unstable();
        neighbors.dedup();
    });
    let bvh = BoundingVolumeHierarchy::from(&surface);
    let mut edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(index, quad)| {
        (0..4).for_each(|i| {
            let mut edge = [quad[i], quad[(i + 1) % 4]];
            edge.sort_unstable();
            edge_owners.entry(edge).or_default().push(index);
        })
    });
    let mut rounds = faces.len() + 1;
    loop {
        match attempt(
            &surface,
            quads.as_slice(),
            &faces,
            &edges,
            &curves,
            &triangles,
            &base,
            &adjacency,
            &bvh,
        ) {
            Ok((imprinted, paths, _)) => {
                match coarsen(&core, &quads, &faces, &edges, imprinted, paths, iterations) {
                    Ok((surface, paths, patches)) => {
                        return Ok(Imprint {
                            core,
                            surface,
                            quads,
                            faces,
                            edges,
                            paths,
                            patches,
                        });
                    }
                    Err(Retry::Fatal(error)) => return Err(error),
                    Err(Retry::Merge(failed)) => {
                        if failed.is_empty() {
                            return Err("imprint failed to converge");
                        }
                        merge_network(
                            &mut faces,
                            &mut edges,
                            &mut curves,
                            &edge_owners,
                            &quads,
                            &failed,
                        );
                        rounds -= 1;
                        if rounds == 0 {
                            return Err("imprint failed to converge");
                        }
                        continue;
                    }
                }
            }
            Err(Retry::Fatal(error)) => return Err(error),
            Err(Retry::Merge(failed)) => {
                merge_network(
                    &mut faces,
                    &mut edges,
                    &mut curves,
                    &edge_owners,
                    &quads,
                    &failed,
                );
            }
        }
        rounds -= 1;
        if rounds == 0 {
            return Err("imprint failed to converge");
        }
    }
}

type Attempt = (Mesh<D>, Vec<Vec<usize>>, Vec<Vec<usize>>);

#[allow(clippy::too_many_arguments)]
fn attempt(
    surface: &Mesh<D>,
    quads: &[Vec<usize>],
    faces: &[Vec<usize>],
    edges: &[[usize; 2]],
    curves: &[Vec<Coordinate<D>>],
    triangles: &[[usize; 3]],
    base: &Coordinates<D>,
    adjacency: &[Vec<usize>],
    bvh: &BoundingVolumeHierarchy<D>,
) -> Result<Attempt, Retry> {
    let mut coordinates = base.clone();
    let number_of_vertices = coordinates.len();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let mut corner_positions: HashMap<usize, Coordinate<D>> = HashMap::new();
    edges
        .iter()
        .zip(curves.iter())
        .for_each(|(&[a, b], curve)| {
            corner_positions
                .entry(a)
                .or_insert_with(|| curve[0].clone());
            corner_positions
                .entry(b)
                .or_insert_with(|| curve[curve.len() - 1].clone());
        });
    let mut corner_list: Vec<usize> = corner_positions.keys().copied().collect();
    corner_list.sort_unstable();
    let mut used = vec![false; number_of_vertices];
    let mut snapped: HashMap<usize, usize> = HashMap::new();
    for &corner in &corner_list {
        let position = &corner_positions[&corner];
        let (_, triangle) = bvh
            .closest_point(position, base, &elements)
            .ok_or(Retry::Fatal("empty surface"))?;
        let mut ring: Vec<usize> = triangles[triangle].to_vec();
        let mut best = None;
        for _ in 0..SNAP_RINGS {
            best = ring
                .iter()
                .copied()
                .filter(|&vertex| !used[vertex])
                .min_by(|&x, &y| {
                    (&coordinates[x] - position)
                        .norm()
                        .total_cmp(&(&coordinates[y] - position).norm())
                });
            if best.is_some() {
                break;
            }
            let mut expanded: Vec<usize> = ring
                .iter()
                .flat_map(|&vertex| adjacency[vertex].iter().copied())
                .collect();
            expanded.sort_unstable();
            expanded.dedup();
            ring = expanded;
        }
        let vertex = best.ok_or(Retry::Fatal(
            "imprint corner snapping failed; refinement required",
        ))?;
        used[vertex] = true;
        snapped.insert(corner, vertex);
    }
    let mut order: Vec<usize> = (0..curves.len()).collect();
    order.sort_by(|&x, &y| {
        let length = |curve: &[Coordinate<D>]| {
            (0..curve.len() - 1)
                .map(|i| (&curve[i + 1] - &curve[i]).norm())
                .sum::<Scalar>()
        };
        length(&curves[x]).total_cmp(&length(&curves[y]))
    });
    let mut paths: Vec<Vec<usize>> = vec![Vec::new(); curves.len()];
    let mut used_edges: HashSet<[usize; 2]> = HashSet::new();
    let mut shared = vec![false; number_of_vertices];
    let mut failed = Vec::new();
    for index in order {
        let [a, b] = edges[index];
        let polyline = &curves[index];
        let (start, goal) = (snapped[&a], snapped[&b]);
        let scale = |vertex: usize| {
            adjacency[vertex]
                .iter()
                .map(|&next| (&coordinates[vertex] - &coordinates[next]).norm())
                .sum::<Scalar>()
                / adjacency[vertex].len() as Scalar
        };
        let step = (scale(start) + scale(goal)) / 2.0;
        let length = (0..polyline.len() - 1)
            .map(|i| (&polyline[i + 1] - &polyline[i]).norm())
            .sum::<Scalar>();
        let cap = ROUTE_CAP * (length + 4.0 * step);
        let path = route(
            start,
            goal,
            polyline,
            &coordinates,
            adjacency,
            &used,
            &used_edges,
            &shared,
            step.max(Scalar::EPSILON),
            cap,
        );
        let Some(path) = path else {
            failed.push(index);
            continue;
        };
        (0..path.len() - 1).for_each(|i| {
            let mut edge = [path[i], path[i + 1]];
            edge.sort_unstable();
            used_edges.insert(edge);
        });
        path.iter()
            .skip(1)
            .take(path.len() - 2)
            .for_each(|&vertex| shared[vertex] = true);
        paths[index] = path;
    }
    if !failed.is_empty() {
        return Err(Retry::Merge(failed));
    }
    corner_list.iter().for_each(|corner| {
        coordinates[snapped[corner]] = corner_positions[corner].clone();
    });
    paths.iter().enumerate().for_each(|(index, path)| {
        path.iter()
            .skip(1)
            .take(path.len() - 2)
            .for_each(|&vertex| {
                coordinates[vertex] = polyline_distance(&coordinates[vertex], &curves[index]).1;
            })
    });
    let mut path_edges: HashMap<[usize; 2], usize> = HashMap::new();
    paths.iter().enumerate().for_each(|(index, path)| {
        (0..path.len() - 1).for_each(|i| {
            let mut edge = [path[i], path[i + 1]];
            edge.sort_unstable();
            path_edges.insert(edge, index);
        })
    });
    let connectivity: Vec<[usize; 3]> = triangles.to_vec();
    let imprinted = Mesh::from((vec![connectivity.into()], coordinates));
    let mut face_of = vec![usize::MAX; quads.len()];
    faces
        .iter()
        .enumerate()
        .for_each(|(face, group)| group.iter().for_each(|&quad| face_of[quad] = face));
    let mut quad_edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(index, quad)| {
        (0..4).for_each(|i| {
            let mut edge = [quad[i], quad[(i + 1) % 4]];
            edge.sort_unstable();
            quad_edge_owners.entry(edge).or_default().push(index);
        })
    });
    let pairs: Vec<[usize; 2]> = edges
        .iter()
        .map(|edge| {
            let owners = &quad_edge_owners[edge];
            [face_of[owners[0]], face_of[owners[1]]]
        })
        .collect();
    match partition_faces(triangles, &path_edges, &pairs, faces.len()) {
        Ok(patches) => Ok((imprinted, paths, patches)),
        Err(unrepresented) => {
            let mut merges = Vec::new();
            unrepresented.into_iter().for_each(|face| {
                let boundary = edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| {
                        let owners = &quad_edge_owners[*edge];
                        face_of[owners[0]] == face || face_of[owners[1]] == face
                    })
                    .max_by(|(x, _), (y, _)| {
                        let length = |index: usize| {
                            let curve = &curves[index];
                            (0..curve.len() - 1)
                                .map(|i| (&curve[i + 1] - &curve[i]).norm())
                                .sum::<Scalar>()
                        };
                        length(*x).total_cmp(&length(*y))
                    })
                    .map(|(index, _)| index);
                if let Some(index) = boundary {
                    merges.push(index);
                }
            });
            if merges.is_empty() {
                return Err(Retry::Fatal(
                    "imprinted regions do not match network faces; refinement required",
                ));
            }
            Err(Retry::Merge(merges))
        }
    }
}

type Coarsened = (Mesh<D>, Vec<Vec<usize>>, Vec<Vec<usize>>);

fn coarsen(
    core: &Mesh<D>,
    quads: &[Vec<usize>],
    faces: &[Vec<usize>],
    edges: &[[usize; 2]],
    surface: Mesh<D>,
    paths: Vec<Vec<usize>>,
    iterations: usize,
) -> Result<Coarsened, Retry> {
    let sizing = QuadSizing::new(core, quads).map_err(Retry::Fatal)?;
    let mut connectivity: Vec<[usize; 3]> = surface
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| from_fn(|i| triangle[i]))
        .collect();
    let mut coordinates = surface.coordinates().clone();
    let mut labels = vec![Label::Free; coordinates.len()];
    let mut edge_map: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    let mut geometry: Vec<Vec<Coordinate<D>>> = Vec::with_capacity(paths.len());
    let mut corner_positions: HashMap<usize, Coordinate<D>> = HashMap::new();
    paths.iter().enumerate().for_each(|(index, path)| {
        geometry.push(
            path.iter()
                .map(|&vertex| coordinates[vertex].clone())
                .collect(),
        );
        path.iter()
            .skip(1)
            .take(path.len() - 2)
            .for_each(|&vertex| labels[vertex] = Label::Curve(index));
        (0..path.len() - 1).for_each(|i| {
            let (u, v) = (path[i], path[i + 1]);
            let key = if u < v { (u, v) } else { (v, u) };
            edge_map.insert(key, index);
        });
    });
    paths.iter().zip(edges.iter()).for_each(|(path, &[a, b])| {
        labels[path[0]] = Label::Corner;
        labels[*path.last().unwrap()] = Label::Corner;
        corner_positions
            .entry(a)
            .or_insert_with(|| coordinates[path[0]].clone());
        corner_positions
            .entry(b)
            .or_insert_with(|| coordinates[*path.last().unwrap()].clone());
    });
    let mut constraints = Constraints {
        labels,
        edges: edge_map,
        curves: geometry,
    };
    remesh(
        &mut connectivity,
        &mut coordinates,
        iterations,
        |_, points, _| sizing.target_lengths(points, 1.0),
        Some(&mut constraints),
    )
    .map_err(Retry::Fatal)?;
    let mut chains: Vec<Vec<[usize; 2]>> = vec![Vec::new(); paths.len()];
    constraints
        .edges
        .iter()
        .for_each(|(&(u, v), &index)| chains[index].push([u, v]));
    let mut recovered = Vec::with_capacity(paths.len());
    for (index, segments) in chains.iter().enumerate() {
        if segments.is_empty() {
            return Err(Retry::Fatal("curve lost during constrained remesh"));
        }
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        segments.iter().for_each(|&[u, v]| {
            adjacency.entry(u).or_default().push(v);
            adjacency.entry(v).or_default().push(u);
        });
        let mut endpoints: Vec<usize> = adjacency
            .iter()
            .filter(|(_, neighbors)| neighbors.len() == 1)
            .map(|(&vertex, _)| vertex)
            .collect();
        if endpoints.len() != 2 {
            return Err(Retry::Fatal("curve tangled during constrained remesh"));
        }
        endpoints.sort_unstable_by(|&x, &y| {
            let anchor = &corner_positions[&edges[index][0]];
            (&coordinates[x] - anchor)
                .norm()
                .total_cmp(&(&coordinates[y] - anchor).norm())
        });
        let start = endpoints[0];
        let mut chain = vec![start];
        let (mut previous, mut current) = (start, adjacency[&start][0]);
        loop {
            chain.push(current);
            let Some(&next) = adjacency[&current]
                .iter()
                .find(|&&candidate| candidate != previous)
            else {
                break;
            };
            previous = current;
            current = next;
        }
        if chain.len() != adjacency.len() {
            return Err(Retry::Fatal("curve tangled during constrained remesh"));
        }
        recovered.push(chain);
    }
    let path_edges: HashMap<[usize; 2], usize> = constraints
        .edges
        .iter()
        .map(|(&(u, v), &index)| ([u.min(v), u.max(v)], index))
        .collect();
    let mut face_of = vec![usize::MAX; quads.len()];
    faces
        .iter()
        .enumerate()
        .for_each(|(face, group)| group.iter().for_each(|&quad| face_of[quad] = face));
    let mut quad_edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(index, quad)| {
        (0..4).for_each(|i| {
            let mut edge = [quad[i], quad[(i + 1) % 4]];
            edge.sort_unstable();
            quad_edge_owners.entry(edge).or_default().push(index);
        })
    });
    let pairs: Vec<[usize; 2]> = edges
        .iter()
        .map(|edge| {
            let owners = &quad_edge_owners[edge];
            [face_of[owners[0]], face_of[owners[1]]]
        })
        .collect();
    let patches = match partition_faces(&connectivity, &path_edges, &pairs, faces.len()) {
        Ok(patches) => patches,
        Err(unrepresented) => {
            let mut merges = Vec::new();
            unrepresented.into_iter().for_each(|face| {
                let boundary = pairs
                    .iter()
                    .enumerate()
                    .filter(|(_, pair)| pair.contains(&face))
                    .max_by(|(x, _), (y, _)| {
                        let length = |index: usize| {
                            let curve = &constraints.curves[index];
                            (0..curve.len() - 1)
                                .map(|i| (&curve[i + 1] - &curve[i]).norm())
                                .sum::<Scalar>()
                        };
                        length(*x).total_cmp(&length(*y))
                    })
                    .map(|(index, _)| index);
                if let Some(index) = boundary {
                    merges.push(index);
                }
            });
            return Err(Retry::Merge(merges));
        }
    };
    let coarse = Mesh::from((vec![connectivity.into()], coordinates));
    Ok((coarse, recovered, patches))
}
fn partition_faces(
    triangles: &[[usize; 3]],
    path_edges: &HashMap<[usize; 2], usize>,
    pairs: &[[usize; 2]],
    number_of_faces: usize,
) -> Result<Vec<Vec<usize>>, Vec<usize>> {
    let mut edge_triangles: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    triangles.iter().enumerate().for_each(|(index, triangle)| {
        (0..3).for_each(|i| {
            let mut edge = [triangle[i], triangle[(i + 1) % 3]];
            edge.sort_unstable();
            edge_triangles.entry(edge).or_default().push(index);
        })
    });
    let mut component = vec![usize::MAX; triangles.len()];
    let mut count = 0;
    (0..triangles.len()).for_each(|seed| {
        if component[seed] != usize::MAX {
            return;
        }
        let mut stack = vec![seed];
        component[seed] = count;
        while let Some(triangle) = stack.pop() {
            (0..3).for_each(|i| {
                let mut edge = [triangles[triangle][i], triangles[triangle][(i + 1) % 3]];
                edge.sort_unstable();
                if path_edges.contains_key(&edge) {
                    return;
                }
                edge_triangles[&edge].iter().for_each(|&neighbor| {
                    if component[neighbor] == usize::MAX {
                        component[neighbor] = count;
                        stack.push(neighbor);
                    }
                });
            })
        }
        count += 1;
    });
    let mut votes: Vec<HashMap<usize, usize>> = vec![HashMap::new(); count];
    path_edges.iter().for_each(|(edge, &curve)| {
        edge_triangles[edge].iter().for_each(|&triangle| {
            pairs[curve].iter().for_each(|&face| {
                *votes[component[triangle]].entry(face).or_default() += 1;
            })
        })
    });
    let claimed: Vec<usize> = votes
        .iter()
        .map(|tally| {
            let mut ranked: Vec<(usize, usize)> =
                tally.iter().map(|(&face, &count)| (count, face)).collect();
            ranked.sort_unstable_by_key(|&(count, face)| (Reverse(count), face));
            match ranked.as_slice() {
                [(top, face), rest @ ..] if rest.is_empty() || rest[0].0 < *top => *face,
                _ => usize::MAX,
            }
        })
        .collect();
    let mut represented = vec![false; number_of_faces];
    claimed.iter().for_each(|&face| {
        if face != usize::MAX {
            represented[face] = true;
        }
    });
    if represented.iter().any(|&seen| !seen) {
        return Err(represented
            .iter()
            .enumerate()
            .filter(|entry| !*entry.1)
            .map(|(face, _)| face)
            .collect());
    }
    let mut labels: Vec<usize> = component.iter().map(|&region| claimed[region]).collect();
    loop {
        let mut changes: Vec<(usize, usize)> = Vec::new();
        labels.iter().enumerate().for_each(|(index, &label)| {
            if label != usize::MAX {
                return;
            }
            let mut counts: HashMap<usize, usize> = HashMap::new();
            (0..3).for_each(|i| {
                let mut edge = [triangles[index][i], triangles[index][(i + 1) % 3]];
                edge.sort_unstable();
                edge_triangles[&edge].iter().for_each(|&neighbor| {
                    if neighbor != index && labels[neighbor] != usize::MAX {
                        *counts.entry(labels[neighbor]).or_default() += 1;
                    }
                });
            });
            if let Some(face) = counts
                .into_iter()
                .min_by(|x, y| (Reverse(x.1), x.0).cmp(&(Reverse(y.1), y.0)))
                .map(|(face, _)| face)
            {
                changes.push((index, face));
            }
        });
        if changes.is_empty() {
            break;
        }
        changes
            .into_iter()
            .for_each(|(index, face)| labels[index] = face);
    }
    if labels.contains(&usize::MAX) {
        return Err(Vec::new());
    }
    let mut patches: Vec<Vec<usize>> = vec![Vec::new(); number_of_faces];
    labels
        .iter()
        .enumerate()
        .for_each(|(index, &face)| patches[face].push(index));
    Ok(patches)
}
#[allow(clippy::too_many_arguments)]
fn route(
    start: usize,
    goal: usize,
    polyline: &[Coordinate<D>],
    coordinates: &Coordinates<D>,
    adjacency: &[Vec<usize>],
    used: &[bool],
    used_edges: &HashSet<[usize; 2]>,
    shared: &[bool],
    step: Scalar,
    cap: Scalar,
) -> Option<Vec<usize>> {
    let mut best: HashMap<usize, Scalar> = HashMap::new();
    let mut previous: HashMap<usize, usize> = HashMap::new();
    let mut heap = BinaryHeap::new();
    best.insert(start, 0.0);
    heap.push(Reverse((Priority(0.0), start)));
    while let Some(Reverse((Priority(distance), vertex))) = heap.pop() {
        if vertex == goal {
            let mut path = vec![goal];
            let mut current = goal;
            while current != start {
                current = previous[&current];
                path.push(current);
            }
            path.reverse();
            return Some(path);
        }
        if distance > best[&vertex] || distance > cap {
            continue;
        }
        for &next in &adjacency[vertex] {
            if used[next] && next != goal {
                continue;
            }
            let key = if vertex < next {
                [vertex, next]
            } else {
                [next, vertex]
            };
            if used_edges.contains(&key) {
                continue;
            }
            let stray = polyline_distance(&coordinates[next], polyline).0 / step;
            let crowding = if shared[next] { SHARE_PENALTY } else { 0.0 };
            let advanced = distance
                + (&coordinates[vertex] - &coordinates[next]).norm()
                    * (1.0 + ROUTE_PENALTY * stray * stray + crowding);
            if best.get(&next).is_none_or(|&known| advanced < known) {
                best.insert(next, advanced);
                previous.insert(next, vertex);
                heap.push(Reverse((Priority(advanced), next)));
            }
        }
    }
    None
}

struct QuadSizing {
    bvh: BoundingVolumeHierarchy<D>,
    coordinates: Coordinates<D>,
    triangles: Vec<[usize; 3]>,
    lengths: Vec<Scalar>,
}

impl QuadSizing {
    fn new(core: &Mesh<D>, quads: &[Vec<usize>]) -> Result<Self, &'static str> {
        let coordinates = core.coordinates();
        if quads.iter().any(|quad| quad.len() != 4) {
            return Err("fitted_core_and_surface requires an all-hexahedral trimmed core");
        }
        let lengths: Vec<Scalar> = quads
            .iter()
            .map(|quad| {
                (0..4)
                    .map(|i| (&coordinates[quad[i]] - &coordinates[quad[(i + 1) % 4]]).norm())
                    .sum::<Scalar>()
                    / 4.0
            })
            .collect();
        let triangles: Vec<[usize; 3]> = quads
            .iter()
            .flat_map(|quad| [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
            .collect();
        let mesh: Mesh<D> = (
            vec![Connectivity::Triangular(triangles.clone().into())],
            coordinates.clone(),
        )
            .into();
        let bvh = BoundingVolumeHierarchy::from(&mesh);
        Ok(Self {
            bvh,
            coordinates: coordinates.clone(),
            triangles,
            lengths,
        })
    }
    fn nearest_quads(&self, points: &Coordinates<D>) -> Vec<Option<usize>> {
        let elements: Vec<&[usize]> = self.triangles.iter().map(|t| t.as_slice()).collect();
        let number_of_points = points.len();
        let mut nearest = vec![None; number_of_points];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_points.div_ceil(threads).max(1);
        scope(|scope| {
            let (bvh, coordinates, elements) = (&self.bvh, &self.coordinates, &elements);
            nearest
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, out)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        out.iter_mut().enumerate().for_each(|(local, slot)| {
                            let point = &points[offset + local];
                            *slot = bvh
                                .closest_point(point, coordinates, elements)
                                .map(|(_, triangle)| triangle / 2);
                        });
                    });
                });
        });
        nearest
    }
    fn target_lengths(&self, points: &Coordinates<D>, factor: Scalar) -> Vec<Scalar> {
        self.nearest_quads(points)
            .into_iter()
            .map(|quad| quad.map_or(Scalar::INFINITY, |quad| self.lengths[quad] * factor))
            .collect()
    }
}
