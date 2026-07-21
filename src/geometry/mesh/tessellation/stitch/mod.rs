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
    collections::{HashMap, HashSet},
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;

pub struct Patches {
    pub quad_root: Vec<usize>,
    pub triangles: Vec<Vec<usize>>,
}

pub struct Stitch {
    pub core: Mesh<D>,
    pub surface: Mesh<D>,
    pub quads: Vec<Vec<usize>>,
    pub patches: Patches,
    pub walls: Vec<Vec<[usize; 3]>>,
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
        let patches = assign_patches(&quads, &sizing, &surface)?;
        let walls = loft_walls(&quads, &core, &surface, &patches)?;
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
    fn target_lengths(&self, points: &Coordinates<D>) -> Vec<Scalar> {
        self.nearest_quads(points)
            .into_iter()
            .map(|quad| quad.map_or(Scalar::INFINITY, |quad| self.lengths[quad]))
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
    let mut triangles: Vec<Vec<usize>> = vec![Vec::new(); number_of_quads];
    nearest
        .into_iter()
        .enumerate()
        .try_for_each(|(triangle, quad)| {
            triangles[quad.ok_or("surface triangle has no nearby quad")?].push(triangle);
            Ok::<(), &'static str>(())
        })?;
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

fn validate_patch_connectivity(
    surface_triangles: &[&[usize]],
    patches: &[Vec<usize>],
) -> Result<(), &'static str> {
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

pub fn loft_walls(
    quads: &[Vec<usize>],
    core: &Mesh<D>,
    surface: &Mesh<D>,
    patches: &Patches,
) -> Result<Vec<Vec<[usize; 3]>>, &'static str> {
    let number_of_quads = quads.len();
    let core_coordinates = core.coordinates();
    let surface_coordinates = surface.coordinates();
    let surface_triangles: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let offset = core.number_of_nodes();
    let mut combined = core_coordinates.clone();
    (0..surface_coordinates.len())
        .for_each(|node| combined.push(surface_coordinates[node].clone()));
    (0..number_of_quads)
        .filter(|&quad| patches.quad_root[quad] == quad)
        .map(|root| {
            let cluster: Vec<&Vec<usize>> = (0..number_of_quads)
                .filter(|&quad| patches.quad_root[quad] == root)
                .map(|quad| &quads[quad])
                .collect();
            let inner = boundary_loop(cluster.iter().map(|quad| quad.as_slice()), 4)?;
            let mut outer = boundary_loop(
                patches.triangles[root]
                    .iter()
                    .map(|&triangle| surface_triangles[triangle]),
                3,
            )?;
            if loop_normal(&inner, core_coordinates) * loop_normal(&outer, surface_coordinates)
                < 0.0
            {
                outer.reverse();
            }
            let outer: Vec<usize> = outer.iter().map(|&node| node + offset).collect();
            Ok(loft(&inner, &outer, &combined))
        })
        .collect::<Result<Vec<_>, &'static str>>()
        .map(|walls| {
            let mut all = vec![Vec::new(); number_of_quads];
            (0..number_of_quads)
                .filter(|&quad| patches.quad_root[quad] == quad)
                .zip(walls)
                .for_each(|(root, wall)| all[root] = wall);
            all
        })
        .and_then(|walls| {
            validate_seam_consistency(quads, &patches.quad_root, &walls)?;
            Ok(walls)
        })
}

fn validate_seam_consistency(
    quads: &[Vec<usize>],
    quad_root: &[usize],
    walls: &[Vec<[usize; 3]>],
) -> Result<(), &'static str> {
    let mut edge_owners: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    quads.iter().enumerate().for_each(|(quad, nodes)| {
        (0..4).for_each(|i| {
            let mut edge = [nodes[i], nodes[(i + 1) % 4]];
            edge.sort_unstable();
            edge_owners.entry(edge).or_default().push(quad);
        })
    });
    let apex = |wall: &[[usize; 3]], edge: [usize; 2]| -> Option<usize> {
        let mut found = None;
        for triangle in wall {
            if triangle.contains(&edge[0]) && triangle.contains(&edge[1]) {
                if found.is_some() {
                    return None;
                }
                found = triangle
                    .iter()
                    .copied()
                    .find(|&node| node != edge[0] && node != edge[1]);
            }
        }
        found
    };
    edge_owners
        .into_iter()
        .filter(|(_, owners)| owners.len() == 2)
        .try_for_each(|(edge, owners)| {
            let (ra, rb) = (quad_root[owners[0]], quad_root[owners[1]]);
            if ra == rb {
                return Ok(());
            }
            match (apex(&walls[ra], edge), apex(&walls[rb], edge)) {
                (Some(a), Some(b)) if a == b => Ok(()),
                _ => Err("wall seam between neighboring patches does not conform"),
            }
        })
}

fn boundary_loop<'a>(
    faces: impl Iterator<Item = &'a [usize]>,
    sides: usize,
) -> Result<Vec<usize>, &'static str> {
    let mut directed: HashMap<(usize, usize), u8> = HashMap::new();
    faces.for_each(|face| {
        (0..sides).for_each(|i| {
            *directed
                .entry((face[i], face[(i + 1) % sides]))
                .or_insert(0) += 1;
        })
    });
    let boundary: Vec<(usize, usize)> = directed
        .keys()
        .filter(|&&(a, b)| !directed.contains_key(&(b, a)))
        .copied()
        .collect();
    let next: HashMap<usize, usize> = boundary.iter().copied().collect();
    if next.len() != boundary.len() {
        return Err("boundary is not a simple loop");
    }
    let Some(&(start, _)) = boundary.first() else {
        return Err("empty boundary");
    };
    let mut nodes = vec![start];
    let mut current = next[&start];
    while current != start {
        nodes.push(current);
        current = *next.get(&current).ok_or("open boundary")?;
    }
    if nodes.len() != boundary.len() {
        return Err("boundary has multiple loops");
    }
    Ok(nodes)
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
