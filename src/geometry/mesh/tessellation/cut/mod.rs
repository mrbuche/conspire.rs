#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinatesRef,
        bbox::BoundingBox,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::collections::{HashMap, hash_map::Entry};

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    Inside,
    Cut,
    Outside,
}

fn contained(mesh: &Mesh<D>, classes: &[Class]) -> bool {
    let mut faces: HashMap<Vec<usize>, (usize, u8)> = HashMap::new();
    let mut offset = 0;
    mesh.iter().for_each(|block| {
        let local_faces = block.local_faces();
        block.iter().enumerate().for_each(|(local, element)| {
            local_faces.iter().for_each(|face| {
                let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                key.sort_unstable();
                faces
                    .entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((offset + local, 1));
            })
        });
        offset += block.number_of_elements();
    });
    faces
        .into_values()
        .all(|(owner, count)| count != 1 || classes[owner] == Class::Outside)
}

impl Tessellation {
    pub fn cut(&self, balancing: Balancing, scale: Scalar) -> Result<Mesh<D>, &'static str> {
        let mut octree =
            Octree::<u16, usize>::from_features(self, scale, CurvatureSizing::default(), PADDING);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mesh = octree.dualize();
        let classes = self.classify(&mesh);
        if !contained(&mesh, &classes) {
            return Err("tessellation is not contained within the dual mesh");
        }
        let (connectivities, coordinates) = mesh.into();
        let hexes: Vec<[usize; 8]> = Vec::try_from(connectivities)?;
        let mut blocks: [Vec<[usize; 8]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        hexes
            .into_iter()
            .zip(classes)
            .for_each(|(hex, class)| blocks[class as usize].push(hex));
        Ok((
            blocks
                .into_iter()
                .filter(|block| !block.is_empty())
                .map(|block| Connectivity::Hexahedral(block.into()))
                .collect::<Vec<Connectivity>>(),
            coordinates,
        )
            .into())
    }
    pub fn classify(&self, mesh: &Mesh<D>) -> Vec<Class> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let number_of_elements = mesh.number_of_elements();
        let mut cut = vec![false; number_of_elements];
        mesh.iter()
            .flat_map(|block| block.iter())
            .zip(cut.iter_mut())
            .for_each(|(element, flag)| {
                let bbox: BoundingBox<D> = element
                    .iter()
                    .map(|&node| &coordinates[node])
                    .collect::<CoordinatesRef<'_, D>>()
                    .into();
                *flag = bvh.overlapping(&bbox).into_iter().any(|triangle| {
                    let nodes = elements[triangle];
                    bbox.overlaps_triangle(
                        &surface_coordinates[nodes[0]],
                        &surface_coordinates[nodes[1]],
                        &surface_coordinates[nodes[2]],
                    )
                })
            });
        let mut faces = HashMap::new();
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); number_of_elements];
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().enumerate().for_each(|(local, element)| {
                let index = offset + local;
                if !cut[index] {
                    local_faces.iter().for_each(|face| {
                        let mut key: Vec<usize> =
                            face.iter().map(|&local| element[local]).collect();
                        key.sort_unstable();
                        match faces.entry(key) {
                            Entry::Occupied(entry) => {
                                let other = *entry.get();
                                neighbors[index].push(other);
                                neighbors[other].push(index);
                            }
                            Entry::Vacant(slot) => {
                                slot.insert(index);
                            }
                        }
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let contains = |point: &Coordinate<D>| {
            directions
                .iter()
                .find_map(|direction| {
                    let ray = (point.clone(), direction.clone()).into();
                    match bvh.intersect(&ray, surface_coordinates, &elements) {
                        None => Some(false),
                        Some(hit) => {
                            let normal = normals[hit.index()];
                            let cosine = (direction * normal) / normal.norm();
                            (cosine.abs() > GRAZING_TOLERANCE).then_some(cosine > 0.0)
                        }
                    }
                })
                .unwrap_or(false)
        };
        let centroids = mesh.centroids();
        let mut classes: Vec<Class> = cut
            .iter()
            .map(|&flag| if flag { Class::Cut } else { Class::Outside })
            .collect();
        let mut visited = cut;
        let mut stack = Vec::new();
        (0..number_of_elements).for_each(|seed| {
            if !visited[seed] {
                let class = if contains(&centroids[seed]) {
                    Class::Inside
                } else {
                    Class::Outside
                };
                visited[seed] = true;
                stack.push(seed);
                while let Some(index) = stack.pop() {
                    classes[index] = class;
                    neighbors[index].iter().for_each(|&next| {
                        if !visited[next] {
                            visited[next] = true;
                            stack.push(next);
                        }
                    })
                }
            }
        });
        classes
    }
}
