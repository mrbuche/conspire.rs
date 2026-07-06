#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::BoundingBox,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, hash_map::Entry},
};

const CROSSING_TOLERANCE: Scalar = 1.0e-8;
const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    Inside,
    Cut,
    Outside,
}

pub struct Tables {
    signs: HashMap<usize, bool>,
    crossings: HashMap<[usize; 2], Coordinate<D>>,
    segments: HashMap<Vec<usize>, Vec<[[usize; 2]; 2]>>,
}

impl Tables {
    pub fn signs(&self) -> &HashMap<usize, bool> {
        &self.signs
    }
    pub fn crossings(&self) -> &HashMap<[usize; 2], Coordinate<D>> {
        &self.crossings
    }
    pub fn segments(&self) -> &HashMap<Vec<usize>, Vec<[[usize; 2]; 2]>> {
        &self.segments
    }
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
        let centroids = mesh.centroids();
        let mut classes: Vec<Class> = cut
            .iter()
            .map(|&flag| if flag { Class::Cut } else { Class::Outside })
            .collect();
        let mut visited = cut;
        let mut stack = Vec::new();
        (0..number_of_elements).for_each(|seed| {
            if !visited[seed] {
                let class = if self.encloses(
                    &centroids[seed],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                ) {
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
    pub fn tables(&self, mesh: &Mesh<D>, classes: &[Class]) -> Result<Tables, &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut edges = HashSet::new();
        let mut face_loops = HashMap::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    EDGES.iter().for_each(|&[a, b]| {
                        let mut key = [element[a], element[b]];
                        key.sort_unstable();
                        edges.insert(key);
                    });
                    local_faces.iter().for_each(|face| {
                        let corners: [usize; 4] = from_fn(|i| element[face[i]]);
                        let mut key = corners.to_vec();
                        key.sort_unstable();
                        face_loops.entry(key).or_insert(corners);
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let mut signs = HashMap::new();
        edges.iter().flatten().for_each(|&node| {
            signs.entry(node).or_insert_with(|| {
                self.encloses(
                    &coordinates[node],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                )
            });
        });
        let mut crossings = HashMap::new();
        edges.iter().try_for_each(|&[a, b]| {
            let span = &coordinates[b] - &coordinates[a];
            let length = span.norm();
            let hit = bvh.intersect(
                &(coordinates[a].clone(), span.clone()).into(),
                surface_coordinates,
                &elements,
            );
            if signs[&a] != signs[&b] {
                let near = hit.ok_or("crossing missing on a sign-change edge")?;
                let far = bvh
                    .intersect(
                        &(coordinates[b].clone(), &coordinates[a] - &coordinates[b]).into(),
                        surface_coordinates,
                        &elements,
                    )
                    .ok_or("crossing missing on a sign-change edge")?;
                if (length - near.distance() - far.distance()).abs()
                    > CROSSING_TOLERANCE.max(GRAZING_TOLERANCE * length)
                {
                    return Err("edge crosses the tessellation more than once");
                }
                crossings.insert(
                    [a, b],
                    &coordinates[a] + &(&span * (near.distance() / length)),
                );
            } else if hit.is_some_and(|near| near.distance() <= length) {
                return Err("edge crosses the tessellation more than once");
            }
            Ok(())
        })?;
        let mut segments = HashMap::new();
        face_loops.into_iter().try_for_each(|(key, [a, b, c, d])| {
            let loop_edges: [[usize; 2]; 4] = [[a, b], [b, c], [c, d], [d, a]].map(|mut edge| {
                edge.sort_unstable();
                edge
            });
            let crossed: Vec<usize> = (0..4)
                .filter(|&i| crossings.contains_key(&loop_edges[i]))
                .collect();
            match crossed[..] {
                [] => Ok(()),
                [first, second] => {
                    segments.insert(key, vec![[loop_edges[first], loop_edges[second]]]);
                    Ok(())
                }
                [..] if crossed.len() == 4 => {
                    let center = [a, b, c, d]
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .sum::<Coordinate<D>>()
                        / 4.0;
                    let pairs = if self.encloses(
                        &center,
                        surface_coordinates,
                        &elements,
                        &normals,
                        &directions,
                    ) == signs[&a]
                    {
                        vec![
                            [loop_edges[0], loop_edges[1]],
                            [loop_edges[2], loop_edges[3]],
                        ]
                    } else {
                        vec![
                            [loop_edges[3], loop_edges[0]],
                            [loop_edges[1], loop_edges[2]],
                        ]
                    };
                    segments.insert(key, pairs);
                    Ok(())
                }
                _ => Err("inconsistent crossings around a face"),
            }
        })?;
        Ok(Tables {
            signs,
            crossings,
            segments,
        })
    }
    fn encloses(
        &self,
        point: &Coordinate<D>,
        surface_coordinates: &Coordinates<D>,
        elements: &[&[usize]],
        normals: &[&Coordinate<D>],
        directions: &[Coordinate<D>; 3],
    ) -> bool {
        directions
            .iter()
            .find_map(|direction| {
                let ray = (point.clone(), direction.clone()).into();
                match self.bvh().intersect(&ray, surface_coordinates, elements) {
                    None => Some(false),
                    Some(hit) => {
                        let normal = normals[hit.index()];
                        let cosine = (direction * normal) / normal.norm();
                        (cosine.abs() > GRAZING_TOLERANCE).then_some(cosine > 0.0)
                    }
                }
            })
            .unwrap_or(false)
    }
}
