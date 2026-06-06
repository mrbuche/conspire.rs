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
        ntree::{Balance, Balancing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, hash_map::Entry},
    thread::{available_parallelism, scope},
};

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];
impl Tessellation {
    pub fn dualize(&self, scale: Scalar) -> Result<Mesh<D>, &'static str> {
        let (mut octree, bvh) = Octree::<u16, usize>::from_sdf(self, scale);
        octree.equilibrate(Balancing::Strong, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh, &bvh);
        self.project_boundary(mesh, &bvh)
    }
    fn project_boundary(
        &self,
        mesh: Mesh<D>,
        bvh: &BoundingVolumeHierarchy<D>,
    ) -> Result<Mesh<D>, &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let boundary = mesh.exterior_faces();
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
    fn trim(&self, mesh: &mut Mesh<D>, bvh: &BoundingVolumeHierarchy<D>) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let coordinates = mesh.coordinates();
        let number_of_nodes = coordinates.len();
        let mut inside = vec![false; number_of_nodes];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_nodes.div_ceil(threads).max(1);
        scope(|scope| {
            let (elements, normals, directions) = (&elements, &normals, &directions);
            inside
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, flags)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        flags.iter_mut().enumerate().for_each(|(local, flag)| {
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
                        });
                    });
                });
        });
        let mut remap = vec![usize::MAX; inside.len()];
        let mut coordinates = Coordinates::new();
        let mut connectivity = Vec::new();
        mesh.iter()
            .flatten()
            .filter(|element| element.iter().all(|&node| inside[node]))
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
