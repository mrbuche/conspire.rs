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
        ntree::{Balance, Balancing, Dualization, Octree, OrthotreeError, Pairing},
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::HashMap,
    thread::{available_parallelism, scope},
};

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];
const HEX_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

impl Tessellation {
    pub fn dualize(&self, scale: Scalar) -> Result<Mesh<D>, OrthotreeError> {
        let (mut octree, bvh) = Octree::<u16, usize>::from_sdf(self, scale);
        octree.equilibrate(Balancing::Strong, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh, &bvh);
        Ok(self.project_boundary(mesh, &bvh))
    }
    fn project_boundary(&self, mesh: Mesh<D>, bvh: &BoundingVolumeHierarchy<D>) -> Mesh<D> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let (connectivities, mut coordinates) = mesh.into();
        let mut connectivity: Vec<[usize; 8]> = connectivities
            .members()
            .iter()
            .flatten()
            .map(|hex| from_fn(|i| hex[i]))
            .collect();
        let mut faces = HashMap::new();
        connectivity.iter().for_each(|hex| {
            HEX_FACES.iter().for_each(|face| {
                let quad: [usize; 4] = from_fn(|k| hex[face[k]]);
                let mut key = quad;
                key.sort_unstable();
                faces
                    .entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((quad, 1));
            })
        });
        let boundary: Vec<[usize; 4]> = faces
            .into_values()
            .filter_map(|(quad, count)| (count == 1).then_some(quad))
            .collect();
        let mut projection = HashMap::new();
        boundary.iter().flatten().for_each(|&node| {
            projection.entry(node).or_insert_with(|| {
                let query = &coordinates[node];
                let point = bvh
                    .closest_point(query, surface_coordinates, &elements)
                    .expect("empty tessellation")
                    .0;
                let index = coordinates.len();
                coordinates.push(point);
                index
            });
        });
        boundary.iter().for_each(|&[a, b, c, d]| {
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
        (
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into()
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
