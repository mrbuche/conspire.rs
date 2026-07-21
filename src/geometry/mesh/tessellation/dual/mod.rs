#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, hash_map::Entry},
    thread::{available_parallelism, scope},
};

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const TRIM_MARGIN: Scalar = 0.5;
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
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];
impl Tessellation {
    pub fn dualize(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
    ) -> Result<Mesh<D>, &'static str> {
        let mut octree = Octree::<u16, usize>::from_features(self, scale, curvature, 0);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh, self.bvh(), TRIM_MARGIN);
        self.buffer(mesh, self.bvh())
    }
    fn buffer(
        &self,
        mesh: Mesh<D>,
        bvh: &BoundingVolumeHierarchy<D>,
    ) -> Result<Mesh<D>, &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let boundary = mesh.exterior_faces();
        let mut edges = HashMap::new();
        boundary.iter().for_each(|face| {
            (0..face.len()).for_each(|i| {
                let mut edge = [face[i], face[(i + 1) % face.len()]];
                edge.sort_unstable();
                *edges.entry(edge).or_insert(0u8) += 1;
            })
        });
        if edges.values().any(|&count| count != 2) {
            return Err("non-manifold boundary");
        }
        let (connectivities, mut coordinates) = mesh.into();
        let mut connectivity = Vec::try_from(connectivities)?;
        let mut projection = HashMap::new();
        boundary.iter().flatten().try_for_each(|&node| {
            if let Entry::Vacant(slot) = projection.entry(node) {
                let point = bvh
                    .closest_point(&coordinates[node], surface_coordinates, &elements)
                    .ok_or("empty tessellation")?
                    .0;
                slot.insert(coordinates.len());
                coordinates.push(point);
            }
            Ok(())
        })?;
        boundary.iter().for_each(|face| {
            let [a, b, c, d] = [face[0], face[1], face[2], face[3]];
            connectivity.push([
                a,
                b,
                c,
                d,
                projection[&a],
                projection[&b],
                projection[&c],
                projection[&d],
            ])
        });
        Ok((
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into())
    }
    pub(crate) fn trim(
        &self,
        mesh: &mut Mesh<D>,
        bvh: &BoundingVolumeHierarchy<D>,
        margin: Scalar,
    ) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let coordinates = mesh.coordinates();
        let number_of_nodes = coordinates.len();
        let mut inside = vec![false; number_of_nodes];
        let mut clearance = vec![0.0; number_of_nodes];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_nodes.div_ceil(threads).max(1);
        scope(|scope| {
            let (elements, normals, directions) = (&elements, &normals, &directions);
            inside
                .chunks_mut(chunk_size)
                .zip(clearance.chunks_mut(chunk_size))
                .enumerate()
                .for_each(|(chunk, (flags, distances))| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        flags
                            .iter_mut()
                            .zip(distances.iter_mut())
                            .enumerate()
                            .for_each(|(local, (flag, distance))| {
                                let point = &coordinates[offset + local];
                                *flag = directions
                                    .iter()
                                    .find_map(|direction| {
                                        let ray = (point.clone(), direction.clone()).into();
                                        match bvh.intersect(&ray, surface_coordinates, elements) {
                                            None => Some(false),
                                            Some(hit) => {
                                                let normal = normals[hit.index()];
                                                let cosine = (direction * normal) / normal.norm();
                                                (cosine.abs() > GRAZING_TOLERANCE)
                                                    .then_some(cosine > 0.0)
                                            }
                                        }
                                    })
                                    .unwrap_or(false);
                                if *flag
                                    && let Some((closest, _)) =
                                        bvh.closest_point(point, surface_coordinates, elements)
                                {
                                    *distance = (&closest - point).norm();
                                }
                            });
                    });
                });
        });
        let mut remap = vec![usize::MAX; inside.len()];
        let mut coordinates = Coordinates::new();
        let mut connectivity = Vec::new();
        mesh.iter()
            .flatten()
            .filter(|element| {
                element.iter().all(|&node| inside[node]) && {
                    let margin = margin
                        * EDGES
                            .iter()
                            .map(|&[a, b]| {
                                (&mesh.coordinates()[element[a]] - &mesh.coordinates()[element[b]])
                                    .norm()
                            })
                            .fold(Scalar::INFINITY, Scalar::min);
                    element.iter().all(|&node| clearance[node] >= margin)
                }
            })
            .for_each(|element| {
                connectivity.push(from_fn(|i| {
                    let node = element[i];
                    if remap[node] == usize::MAX {
                        remap[node] = coordinates.len();
                        coordinates.push(mesh.coordinates()[node].clone());
                    }
                    remap[node]
                }))
            });
        *mesh = (
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into();
    }
}
