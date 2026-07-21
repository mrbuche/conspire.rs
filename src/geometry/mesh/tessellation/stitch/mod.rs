#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            remesh::triangles::remesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::{
    array::from_fn,
    collections::HashMap,
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;

/// The whole-face assignment of a fitted surface trimesh to the retained
/// boundary quads of a trimmed hex core, used to drive the (not yet built)
/// per-quad loft/zipper stitch.
pub struct Patches {
    /// For each boundary quad, the root quad of its merged cluster (itself,
    /// unless it had no assigned triangles and was absorbed into a
    /// neighbor's cluster).
    pub quad_root: Vec<usize>,
    /// For each boundary quad, the surface triangle indices assigned to it.
    /// Only populated at cluster roots (`quad_root[quad] == quad`); merged
    /// quads have an empty entry here.
    pub triangles: Vec<Vec<usize>>,
}

impl Tessellation {
    /// Builds an independent, quality-controlled triangle mesh sized to match
    /// the boundary quads of the trimmed interior hex mesh, and assigns each
    /// of its triangles (whole-face, never split) to the nearest boundary
    /// quad, as a first step toward stitching the two together (the
    /// polyhedral transition layer itself is not built yet).
    pub fn fitted_surface(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<(Mesh<D>, Mesh<D>, Patches), &'static str> {
        let mut octree = Octree::<u16, usize>::from_features(self, scale, curvature, 0);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mut core = octree.dualize();
        self.trim(&mut core, self.bvh(), STITCH_TRIM_MARGIN);
        let sizing = QuadSizing::new(&core)?;
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
        let surface: Mesh<D> = (vec![connectivity.into()], coordinates).into();
        let patches = assign_patches(&core, &sizing, &surface)?;
        Ok((core, surface, patches))
    }
}

struct QuadSizing {
    bvh: BoundingVolumeHierarchy<D>,
    coordinates: Coordinates<D>,
    triangles: Vec<[usize; 3]>,
    lengths: Vec<Scalar>,
}

impl QuadSizing {
    fn new(core: &Mesh<D>) -> Result<Self, &'static str> {
        let coordinates = core.coordinates();
        let quads = core.exterior_faces();
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
    /// The nearest quad (by index into `core.exterior_faces()`) to each
    /// point, or `None` if the quad set is empty.
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
    core: &Mesh<D>,
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
    let quads = core.exterior_faces();
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
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); number_of_quads];
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
    Ok(Patches {
        quad_root,
        triangles,
    })
}
