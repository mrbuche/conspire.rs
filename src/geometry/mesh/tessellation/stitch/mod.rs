#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            remesh::triangles::remesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{CrossProduct, Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;

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

pub struct Patches {
    pub quad_root: Vec<usize>,
    pub triangles: Vec<Vec<usize>>,
}

pub struct Wall {
    pub pair: [usize; 2],
    pub polygon: Vec<usize>,
}

pub struct Stitch {
    pub core: Mesh<D>,
    pub surface: Mesh<D>,
    pub quads: Vec<Vec<usize>>,
    pub patches: Patches,
    pub walls: Vec<Wall>,
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
            |_, points, _| sizing.target_lengths(points),
        )?;
        let surface = Mesh::from((vec![connectivity.into()], coordinates));
        Ok((core, surface))
    }
    pub fn fitted_surface(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<Stitch, &'static str> {
        let (core, surface) =
            self.fitted_core_and_surface(balancing, scale, curvature, iterations)?;
        let quads = core.exterior_faces();
        let sizing = QuadSizing::new(&core, &quads)?;
        let mut patches = assign_patches(&quads, &sizing, &surface)?;
        let walls = stitch_walls(&quads, &core, &surface, &mut patches)?;
        Ok(Stitch {
            core,
            surface,
            quads,
            patches,
            walls,
        })
    }
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
            return Err("fitted_surface requires an all-hexahedral trimmed core");
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
    fn nearest_quads(&self, points: &Coordinates<D>) -> Vec<Option<(usize, Scalar)>> {
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
                            *slot = bvh.closest_point(point, coordinates, elements).map(
                                |(closest, triangle)| (triangle / 2, (&closest - point).norm()),
                            );
                        });
                    });
                });
        });
        nearest
    }
    fn target_lengths(&self, points: &Coordinates<D>) -> Vec<Scalar> {
        self.nearest_quads(points)
            .into_iter()
            .map(|quad| quad.map_or(Scalar::INFINITY, |(quad, _)| self.lengths[quad]))
            .collect()
    }
}

fn assign_patches(
    quads: &[Vec<usize>],
    sizing: &QuadSizing,
    surface: &Mesh<D>,
) -> Result<Patches, &'static str> {
    let surface_coordinates = surface.coordinates();
    let surface_triangles: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let centroids: Coordinates<D> = surface_triangles
        .iter()
        .map(|triangle| {
            (&surface_coordinates[triangle[0]]
                + &surface_coordinates[triangle[1]]
                + &surface_coordinates[triangle[2]])
                / 3.0
        })
        .collect();
    let nearest = sizing.nearest_quads(&centroids);
    let number_of_quads = quads.len();
    let number_of_triangles = surface_triangles.len();
    let mut seeds: Vec<Option<(usize, Scalar)>> = vec![None; number_of_quads];
    nearest.iter().enumerate().try_for_each(|(triangle, hit)| {
        let (quad, distance) = hit.ok_or("surface triangle has no nearby quad")?;
        if seeds[quad].is_none_or(|(_, best)| distance < best) {
            seeds[quad] = Some((triangle, distance));
        }
        Ok::<(), &'static str>(())
    })?;
    let neighbors = triangle_neighbors(&surface_triangles);
    let mut labels = vec![usize::MAX; number_of_triangles];
    let mut heap = BinaryHeap::new();
    seeds.iter().enumerate().for_each(|(quad, seed)| {
        if let Some((triangle, distance)) = seed {
            heap.push(Reverse((Priority(*distance), *triangle, quad)));
        }
    });
    while let Some(Reverse((Priority(distance), triangle, quad))) = heap.pop() {
        if labels[triangle] != usize::MAX {
            continue;
        }
        labels[triangle] = quad;
        neighbors[triangle].iter().for_each(|&neighbor| {
            if labels[neighbor] == usize::MAX {
                let step = (&centroids[neighbor] - &centroids[triangle]).norm();
                heap.push(Reverse((Priority(distance + step), neighbor, quad)));
            }
        });
    }
    if labels.contains(&usize::MAX) {
        return Err("surface triangle has no nearby quad");
    }
    let mut triangles: Vec<Vec<usize>> = vec![Vec::new(); number_of_quads];
    labels
        .into_iter()
        .enumerate()
        .for_each(|(triangle, quad)| triangles[quad].push(triangle));
    let mut edge_quads: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(quad, nodes)| {
        (0..4).for_each(|i| {
            let mut edge = [nodes[i], nodes[(i + 1) % 4]];
            edge.sort_unstable();
            edge_quads.entry(edge).or_default().push(quad);
        })
    });
    if edge_quads.values().any(|owners| owners.len() != 2) {
        return Err("non-manifold quad boundary");
    }
    let mut neighbors = vec![Vec::new(); number_of_quads];
    edge_quads.values().for_each(|owners| {
        if let [a, b] = owners[..] {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    });
    let has_direct: Vec<bool> = triangles
        .iter()
        .map(|assigned| !assigned.is_empty())
        .collect();
    let mut parent: Vec<usize> = (0..number_of_quads).collect();
    fn find(parent: &mut [usize], node: usize) -> usize {
        if parent[node] == node {
            node
        } else {
            let root = find(parent, parent[node]);
            parent[node] = root;
            root
        }
    }
    let mut cluster_has_direct = has_direct;
    let mut pending: Vec<usize> = (0..number_of_quads)
        .filter(|&quad| !cluster_has_direct[quad])
        .collect();
    let mut changed = true;
    while changed && !pending.is_empty() {
        changed = false;
        pending.retain(|&quad| {
            let root = find(&mut parent, quad);
            if cluster_has_direct[root] {
                return false;
            }
            let Some(&donor) = neighbors[quad]
                .iter()
                .find(|&&neighbor| cluster_has_direct[find(&mut parent, neighbor)])
            else {
                return true;
            };
            let donor_root = find(&mut parent, donor);
            parent[root] = donor_root;
            cluster_has_direct[donor_root] = true;
            changed = true;
            false
        });
    }
    if !pending.is_empty() {
        return Err("an isolated cluster of boundary quads has no nearby surface triangles");
    }
    let quad_root: Vec<usize> = (0..number_of_quads)
        .map(|quad| find(&mut parent, quad))
        .collect();
    (0..number_of_quads).for_each(|quad| {
        let root = quad_root[quad];
        if root != quad {
            let moved = std::mem::take(&mut triangles[quad]);
            triangles[root].extend(moved);
        }
    });
    validate_patch_connectivity(&surface_triangles, &triangles)?;
    Ok(Patches {
        quad_root,
        triangles,
    })
}

fn triangle_neighbors(surface_triangles: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut edge_triangles: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    surface_triangles
        .iter()
        .enumerate()
        .for_each(|(triangle, nodes)| {
            (0..3).for_each(|i| {
                let mut edge = [nodes[i], nodes[(i + 1) % 3]];
                edge.sort_unstable();
                edge_triangles.entry(edge).or_default().push(triangle);
            })
        });
    let mut neighbors = vec![Vec::new(); surface_triangles.len()];
    edge_triangles.values().for_each(|owners| {
        if let [a, b] = owners[..] {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    });
    neighbors
}

fn validate_patch_connectivity(
    surface_triangles: &[&[usize]],
    patches: &[Vec<usize>],
) -> Result<(), &'static str> {
    let neighbors = triangle_neighbors(surface_triangles);
    patches
        .iter()
        .filter(|patch| !patch.is_empty())
        .try_for_each(|patch| {
            let in_patch: HashSet<usize> = patch.iter().copied().collect();
            let mut visited = HashSet::from([patch[0]]);
            let mut stack = vec![patch[0]];
            while let Some(triangle) = stack.pop() {
                neighbors[triangle].iter().for_each(|&neighbor| {
                    if in_patch.contains(&neighbor) && visited.insert(neighbor) {
                        stack.push(neighbor);
                    }
                });
            }
            if visited.len() == patch.len() {
                Ok(())
            } else {
                Err("a patch has disconnected components and requires refinement")
            }
        })
}

type Chains = Vec<(Vec<usize>, bool)>;

type Network = (
    HashMap<[usize; 2], Chains>,
    HashMap<usize, HashSet<usize>>,
    Vec<[usize; 2]>,
);

fn seam_network<'a>(
    faces: impl Iterator<Item = &'a [usize]>,
    labels: &[usize],
    manifold_error: &'static str,
) -> Result<Network, &'static str> {
    let mut edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    faces.enumerate().for_each(|(face, nodes)| {
        let sides = nodes.len();
        (0..sides).for_each(|i| {
            let mut edge = [nodes[i], nodes[(i + 1) % sides]];
            edge.sort_unstable();
            edge_owners.entry(edge).or_default().push(face);
        })
    });
    let mut pair_edges: HashMap<[usize; 2], Vec<[usize; 2]>> = HashMap::new();
    let mut vertex_labels: HashMap<usize, HashSet<usize>> = HashMap::new();
    edge_owners.into_iter().try_for_each(|(edge, owners)| {
        let [a, b] = owners[..] else {
            return Err(manifold_error);
        };
        if labels[a] != labels[b] {
            let mut pair = [labels[a], labels[b]];
            pair.sort_unstable();
            pair_edges.entry(pair).or_default().push(edge);
            edge.iter().for_each(|&vertex| {
                let set = vertex_labels.entry(vertex).or_default();
                set.insert(labels[a]);
                set.insert(labels[b]);
            });
        }
        Ok(())
    })?;
    let corner_labels: HashMap<usize, HashSet<usize>> = vertex_labels
        .into_iter()
        .filter(|(_, set)| set.len() >= 3)
        .collect();
    let corners: HashSet<usize> = corner_labels.keys().copied().collect();
    let mut network = HashMap::new();
    let mut unstitchable = Vec::new();
    pair_edges.into_iter().for_each(|(pair, mut edges)| {
        edges.sort_unstable();
        match seam_chains(&edges, &corners) {
            Ok(chains) => {
                network.insert(pair, chains);
            }
            Err(_) => unstitchable.push(pair),
        }
    });
    Ok((network, corner_labels, unstitchable))
}

fn seam_chains(edges: &[[usize; 2]], corners: &HashSet<usize>) -> Result<Chains, &'static str> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    edges.iter().for_each(|&[a, b]| {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    });
    if adjacency
        .iter()
        .any(|(vertex, neighbors)| !corners.contains(vertex) && neighbors.len() != 2)
    {
        return Err("pinched seam between patches requires refinement");
    }
    let key = |a: usize, b: usize| if a < b { [a, b] } else { [b, a] };
    let mut used: HashSet<[usize; 2]> = HashSet::new();
    let mut chains = Vec::new();
    let mut starts: Vec<usize> = adjacency
        .keys()
        .copied()
        .filter(|vertex| corners.contains(vertex))
        .collect();
    starts.sort_unstable();
    for &start in &starts {
        for first in adjacency[&start].clone() {
            if used.contains(&key(start, first)) {
                continue;
            }
            used.insert(key(start, first));
            let mut nodes = vec![start, first];
            let (mut previous, mut current) = (start, first);
            while !corners.contains(&current) {
                let next = adjacency[&current]
                    .iter()
                    .copied()
                    .find(|&candidate| candidate != previous)
                    .ok_or("open seam chain between patches")?;
                used.insert(key(current, next));
                nodes.push(next);
                previous = current;
                current = next;
            }
            chains.push((nodes, false));
        }
    }
    for &[a, b] in edges {
        if used.contains(&key(a, b)) {
            continue;
        }
        used.insert(key(a, b));
        let mut nodes = vec![a];
        let (mut previous, mut current) = (a, b);
        while current != a {
            nodes.push(current);
            let next = adjacency[&current]
                .iter()
                .copied()
                .find(|&candidate| candidate != previous)
                .ok_or("open seam chain between patches")?;
            used.insert(key(current, next));
            previous = current;
            current = next;
        }
        chains.push((nodes, true));
    }
    Ok(chains)
}

enum Failure {
    Merge(Vec<[usize; 2]>),
    Fatal(&'static str),
}

fn stitch_walls(
    quads: &[Vec<usize>],
    core: &Mesh<D>,
    surface: &Mesh<D>,
    patches: &mut Patches,
) -> Result<Vec<Wall>, &'static str> {
    let mut rounds = patches.quad_root.len() + 1;
    loop {
        match build_walls(quads, core, surface, patches) {
            Ok(walls) => return Ok(walls),
            Err(Failure::Fatal(error)) => return Err(error),
            Err(Failure::Merge(pairs)) => {
                if pairs.is_empty() {
                    return Err("stitching failed to converge");
                }
                merge_patches(patches, &pairs);
            }
        }
        rounds -= 1;
        if rounds == 0 {
            return Err("stitching failed to converge");
        }
    }
}

fn merge_patches(patches: &mut Patches, pairs: &[[usize; 2]]) {
    pairs.iter().for_each(|&[a, b]| {
        let (ra, rb) = (patches.quad_root[a], patches.quad_root[b]);
        if ra == rb {
            return;
        }
        let (keep, drop) = (ra.min(rb), ra.max(rb));
        patches.quad_root.iter_mut().for_each(|root| {
            if *root == drop {
                *root = keep
            }
        });
        let moved = std::mem::take(&mut patches.triangles[drop]);
        patches.triangles[keep].extend(moved);
    });
}

fn build_walls(
    quads: &[Vec<usize>],
    core: &Mesh<D>,
    surface: &Mesh<D>,
    patches: &Patches,
) -> Result<Vec<Wall>, Failure> {
    let surface_triangles: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let mut surface_labels = vec![usize::MAX; surface_triangles.len()];
    patches
        .triangles
        .iter()
        .enumerate()
        .for_each(|(root, list)| {
            list.iter()
                .for_each(|&triangle| surface_labels[triangle] = root)
        });
    let (surface_network, corner_labels, unstitchable) = seam_network(
        surface_triangles.iter().copied(),
        &surface_labels,
        "surface is not a closed manifold",
    )
    .map_err(Failure::Fatal)?;
    if !unstitchable.is_empty() {
        return Err(Failure::Merge(unstitchable));
    }
    let mut edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(quad, nodes)| {
        (0..4).for_each(|i| {
            let mut edge = [nodes[i], nodes[(i + 1) % 4]];
            edge.sort_unstable();
            edge_owners.entry(edge).or_default().push(quad);
        })
    });
    let mut boundary_edges: Vec<([usize; 2], [usize; 2])> = Vec::new();
    for (edge, owners) in &edge_owners {
        let [a, b] = owners[..] else {
            return Err(Failure::Fatal("non-manifold quad boundary"));
        };
        let (ra, rb) = (patches.quad_root[a], patches.quad_root[b]);
        if ra != rb {
            boundary_edges.push((*edge, [ra.min(rb), ra.max(rb)]));
        }
    }
    boundary_edges.sort_unstable();
    let mut root_edges: HashMap<usize, Vec<usize>> = HashMap::new();
    boundary_edges
        .iter()
        .enumerate()
        .for_each(|(index, (_, roots))| {
            roots
                .iter()
                .for_each(|&root| root_edges.entry(root).or_default().push(index))
        });
    let mut vertex_roots: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut root_vertices: HashMap<usize, Vec<usize>> = HashMap::new();
    boundary_edges.iter().for_each(|(edge, roots)| {
        edge.iter().for_each(|&node| {
            roots.iter().for_each(|&root| {
                vertex_roots.entry(node).or_default().insert(root);
                root_vertices.entry(root).or_default().push(node);
            })
        })
    });
    let core_coordinates = core.coordinates();
    let surface_coordinates = surface.coordinates();
    let mut anchors: HashMap<usize, usize> = HashMap::new();
    let mut corner_list: Vec<usize> = corner_labels.keys().copied().collect();
    corner_list.sort_unstable();
    for &corner in &corner_list {
        let labels = &corner_labels[&corner];
        let mut candidates: Vec<usize> = labels
            .iter()
            .flat_map(|root| root_vertices.get(root).cloned().unwrap_or_default())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        if let Some(anchor) = candidates.into_iter().min_by(|&u, &v| {
            let key = |vertex: usize| {
                let overlap = vertex_roots.get(&vertex).map_or(0, |roots| {
                    labels.iter().filter(|label| roots.contains(label)).count()
                });
                let distance = (&core_coordinates[vertex] - &surface_coordinates[corner]).norm();
                (Reverse(overlap), Priority(distance))
            };
            key(u).cmp(&key(v))
        }) {
            anchors.insert(corner, anchor);
        }
    }
    let offset = core.number_of_nodes();
    let mut combined = core_coordinates.clone();
    (0..surface_coordinates.len())
        .for_each(|node| combined.push(surface_coordinates[node].clone()));
    let mut pairs: Vec<[usize; 2]> = surface_network.keys().copied().collect();
    pairs.sort_unstable();
    let mut walls = Vec::new();
    let mut merges = Vec::new();
    for pair in pairs {
        for (chain, closed) in &surface_network[&pair] {
            if *closed {
                let loop_edges: Vec<[usize; 2]> = boundary_edges
                    .iter()
                    .filter(|(_, roots)| roots == &pair)
                    .map(|(edge, _)| *edge)
                    .collect();
                let Some(core_loop) = walk_cycle(&loop_edges) else {
                    merges.push(pair);
                    continue;
                };
                let mut outer = chain.clone();
                if loop_normal(&core_loop, core_coordinates)
                    * loop_normal(&outer, surface_coordinates)
                    < 0.0
                {
                    outer.reverse();
                }
                let outer: Vec<usize> = outer.iter().map(|&node| node + offset).collect();
                loft(&core_loop, &outer, &combined)
                    .into_iter()
                    .for_each(|triangle| {
                        walls.push(Wall {
                            pair,
                            polygon: triangle.to_vec(),
                        })
                    });
            } else {
                let (start, end) = (chain[0], *chain.last().unwrap());
                let (Some(&anchor_start), Some(&anchor_end)) =
                    (anchors.get(&start), anchors.get(&end))
                else {
                    merges.push(pair);
                    continue;
                };
                if start == end {
                    merges.push(pair);
                    continue;
                }
                let path = if anchor_start == anchor_end {
                    vec![anchor_start]
                } else {
                    match core_path(
                        anchor_start,
                        anchor_end,
                        &pair,
                        &boundary_edges,
                        &root_edges,
                        core_coordinates,
                    ) {
                        Some(path) => path,
                        None => {
                            merges.push(pair);
                            continue;
                        }
                    }
                };
                let mut polygon = path;
                polygon.extend(chain.iter().rev().map(|&node| node + offset));
                walls.push(Wall { pair, polygon });
            }
        }
    }
    if !merges.is_empty() {
        return Err(Failure::Merge(merges));
    }
    validate_cells(
        quads,
        patches,
        &surface_triangles,
        &surface_labels,
        &walls,
        offset,
    )?;
    Ok(walls)
}

fn core_path(
    start: usize,
    goal: usize,
    pair: &[usize; 2],
    boundary_edges: &[([usize; 2], [usize; 2])],
    root_edges: &HashMap<usize, Vec<usize>>,
    coordinates: &Coordinates<D>,
) -> Option<Vec<usize>> {
    let mut indices: Vec<usize> = pair
        .iter()
        .flat_map(|root| root_edges.get(root).cloned().unwrap_or_default())
        .collect();
    indices.sort_unstable();
    indices.dedup();
    let shared: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&index| boundary_edges[index].1 == *pair)
        .collect();
    search(start, goal, &shared, boundary_edges, coordinates)
        .or_else(|| search(start, goal, &indices, boundary_edges, coordinates))
}

fn search(
    start: usize,
    goal: usize,
    indices: &[usize],
    boundary_edges: &[([usize; 2], [usize; 2])],
    coordinates: &Coordinates<D>,
) -> Option<Vec<usize>> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    indices.iter().for_each(|&index| {
        let ([a, b], _) = boundary_edges[index];
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    });
    let mut best: HashMap<usize, Scalar> = HashMap::new();
    let mut previous: HashMap<usize, usize> = HashMap::new();
    let mut heap = BinaryHeap::new();
    best.insert(start, 0.0);
    heap.push(Reverse((Priority(0.0), start)));
    while let Some(Reverse((Priority(distance), node))) = heap.pop() {
        if node == goal {
            break;
        }
        if distance > best[&node] {
            continue;
        }
        for &next in adjacency.get(&node).into_iter().flatten() {
            let advanced = distance + (&coordinates[node] - &coordinates[next]).norm();
            if best.get(&next).is_none_or(|&known| advanced < known) {
                best.insert(next, advanced);
                previous.insert(next, node);
                heap.push(Reverse((Priority(advanced), next)));
            }
        }
    }
    if !previous.contains_key(&goal) {
        return None;
    }
    let mut path = vec![goal];
    let mut current = goal;
    while current != start {
        current = previous[&current];
        path.push(current);
    }
    path.reverse();
    Some(path)
}
fn walk_cycle(edges: &[[usize; 2]]) -> Option<Vec<usize>> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    edges.iter().for_each(|&[a, b]| {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    });
    if adjacency.is_empty() || adjacency.values().any(|neighbors| neighbors.len() != 2) {
        return None;
    }
    let start = edges[0][0];
    let mut nodes = vec![start];
    let (mut previous, mut current) = (start, edges[0][1]);
    while current != start {
        nodes.push(current);
        let next = adjacency[&current]
            .iter()
            .copied()
            .find(|&candidate| candidate != previous)?;
        previous = current;
        current = next;
    }
    (nodes.len() == adjacency.len()).then_some(nodes)
}

fn validate_cells(
    quads: &[Vec<usize>],
    patches: &Patches,
    surface_triangles: &[&[usize]],
    surface_labels: &[usize],
    walls: &[Wall],
    offset: usize,
) -> Result<(), Failure> {
    let mut edge_roots: HashMap<[usize; 2], [usize; 2]> = HashMap::new();
    quads.iter().enumerate().for_each(|(quad, nodes)| {
        let root = patches.quad_root[quad];
        (0..4).for_each(|i| {
            let mut edge = [nodes[i], nodes[(i + 1) % 4]];
            edge.sort_unstable();
            let roots = edge_roots.entry(edge).or_insert([root, usize::MAX]);
            if roots[0] != root {
                roots[1] = root;
            }
        })
    });
    let mut faces: Vec<Vec<Vec<usize>>> = vec![Vec::new(); quads.len()];
    quads
        .iter()
        .enumerate()
        .for_each(|(quad, nodes)| faces[patches.quad_root[quad]].push(nodes.clone()));
    surface_triangles
        .iter()
        .enumerate()
        .for_each(|(triangle, nodes)| {
            faces[surface_labels[triangle]].push(nodes.iter().map(|&node| node + offset).collect())
        });
    walls.iter().for_each(|wall| {
        wall.pair
            .iter()
            .for_each(|&root| faces[root].push(wall.polygon.clone()))
    });
    let mut node_labels: HashMap<usize, HashSet<usize>> = HashMap::new();
    surface_triangles
        .iter()
        .enumerate()
        .for_each(|(triangle, nodes)| {
            nodes.iter().for_each(|&node| {
                node_labels
                    .entry(node)
                    .or_default()
                    .insert(surface_labels[triangle]);
            })
        });
    let mut merges = Vec::new();
    for cell in faces.iter().filter(|cell| !cell.is_empty()) {
        let mut counts: HashMap<[usize; 2], usize> = HashMap::new();
        cell.iter().for_each(|face| {
            (0..face.len()).for_each(|i| {
                let mut edge = [face[i], face[(i + 1) % face.len()]];
                edge.sort_unstable();
                *counts.entry(edge).or_default() += 1;
            })
        });
        for (edge, count) in counts {
            if count == 2 {
                continue;
            }
            if edge[0] < offset && edge[1] < offset {
                match edge_roots.get(&edge) {
                    Some(&[a, b]) if b != usize::MAX => {
                        merges.push([a, b]);
                        continue;
                    }
                    _ => {
                        return Err(Failure::Fatal(
                            "transition cell does not close; refinement required",
                        ));
                    }
                }
            }
            let mut labels: Vec<usize> = edge
                .iter()
                .filter(|&&node| node >= offset)
                .flat_map(|&node| {
                    node_labels
                        .get(&(node - offset))
                        .into_iter()
                        .flatten()
                        .copied()
                })
                .collect();
            labels.sort_unstable();
            labels.dedup();
            if labels.len() < 2 {
                return Err(Failure::Fatal(
                    "transition cell does not close; refinement required",
                ));
            }
            (1..labels.len()).for_each(|i| merges.push([labels[0], labels[i]]));
        }
    }
    if merges.is_empty() {
        Ok(())
    } else {
        Err(Failure::Merge(merges))
    }
}

fn loop_normal(nodes: &[usize], coordinates: &Coordinates<D>) -> Coordinate<D> {
    let centroid = nodes
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / nodes.len() as Scalar;
    (0..nodes.len())
        .map(|i| {
            let a = &coordinates[nodes[i]] - &centroid;
            let b = &coordinates[nodes[(i + 1) % nodes.len()]] - &centroid;
            a.cross(&b)
        })
        .sum()
}

fn loft(inner: &[usize], outer: &[usize], coordinates: &Coordinates<D>) -> Vec<[usize; 3]> {
    let (m, n) = (inner.len(), outer.len());
    let cost = |a: usize, b: usize| -> Scalar {
        let delta = &coordinates[a] - &coordinates[b];
        &delta * &delta
    };
    let start = (0..n)
        .min_by(|&i, &j| {
            cost(inner[0], outer[i])
                .partial_cmp(&cost(inner[0], outer[j]))
                .unwrap()
        })
        .unwrap_or(0);
    let outer: Vec<usize> = (0..n).map(|k| outer[(start + k) % n]).collect();
    let mut dp = vec![vec![Scalar::INFINITY; n + 1]; m + 1];
    let mut via_inner = vec![vec![false; n + 1]; m + 1];
    dp[0][0] = 0.0;
    (0..=m).for_each(|i| {
        (0..=n).for_each(|j| {
            if i == 0 && j == 0 {
                return;
            }
            let diagonal = cost(inner[i % m], outer[j % n]);
            let (best, from_inner) = match (i > 0, j > 0) {
                (true, true) if dp[i - 1][j] <= dp[i][j - 1] => (dp[i - 1][j], true),
                (true, true) => (dp[i][j - 1], false),
                (true, false) => (dp[i - 1][j], true),
                (false, true) => (dp[i][j - 1], false),
                (false, false) => unreachable!(),
            };
            dp[i][j] = best + diagonal;
            via_inner[i][j] = from_inner;
        })
    });
    let mut triangles = Vec::with_capacity(m + n);
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if via_inner[i][j] {
            triangles.push([inner[(i - 1) % m], inner[i % m], outer[j % n]]);
            i -= 1;
        } else {
            triangles.push([inner[i % m], outer[(j - 1) % n], outer[j % n]]);
            j -= 1;
        }
    }
    triangles
}
