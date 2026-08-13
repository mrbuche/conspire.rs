#[cfg(test)]
mod test;

use super::{
    CROSSING_TOLERANCE, Class, DIRECTIONS, Sign, Tables, Vertex,
    face::face_cut,
    geometry::dedupe,
    topology::{element_edges, element_faces, face_owners, oriented_element_faces},
};
use crate::{
    geometry::{
        Coordinate, DirectionsRef,
        mesh::{Mesh, tessellation::D, tessellation::Tessellation},
    },
    math::{Quantity, Scalar, Tensor, unit::Length},
};
use std::{array::from_fn, collections::HashMap, collections::HashSet};

pub(super) struct GenericTables {
    signs: HashMap<usize, Sign>,
    crossings: HashMap<[usize; 2], Vec<Coordinate<D>>>,
    faces: HashMap<Vec<usize>, Vec<usize>>,
    segments: HashMap<Vec<usize>, Vec<[Vertex; 2]>>,
}

impl GenericTables {
    pub(super) fn signs(&self) -> &HashMap<usize, Sign> {
        &self.signs
    }
    pub(super) fn crossings(&self) -> &HashMap<[usize; 2], Vec<Coordinate<D>>> {
        &self.crossings
    }
    pub(super) fn faces(&self) -> &HashMap<Vec<usize>, Vec<usize>> {
        &self.faces
    }
    pub(super) fn segments(&self) -> &HashMap<Vec<usize>, Vec<[Vertex; 2]>> {
        &self.segments
    }
}

impl Tessellation {
    #[allow(clippy::type_complexity)]
    fn edges_signs_crossings(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        snapped: &HashSet<usize>,
    ) -> Result<
        (
            HashSet<[usize; 2]>,
            HashMap<usize, Sign>,
            HashMap<[usize; 2], Vec<Coordinate<D>>>,
        ),
        &'static str,
    > {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut edges = HashSet::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    element_edges(&element_faces(block, element))
                        .into_iter()
                        .for_each(|key| {
                            edges.insert(key);
                        });
                }
            });
            offset += block.number_of_elements();
        });
        let mut signs = HashMap::new();
        edges.iter().flatten().for_each(|&node| {
            signs.entry(node).or_insert_with(|| {
                if snapped.contains(&node) {
                    Sign::On
                } else if self.encloses(
                    &coordinates[node],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                ) {
                    Sign::Inside
                } else {
                    Sign::Outside
                }
            });
        });
        let mut crossings = HashMap::<[usize; 2], Vec<Coordinate<D>>>::new();
        edges.iter().try_for_each(|&[a, b]| {
            let span = &coordinates[b] - &coordinates[a];
            let length = span.norm();
            let margin = CROSSING_TOLERANCE.max(length * super::GRAZING_TOLERANCE);
            match (signs[&a], signs[&b]) {
                (Sign::On, Sign::On) => Ok(()),
                (Sign::On, _) | (_, Sign::On) => {
                    let (from, along) = if signs[&a] == Sign::On {
                        (b, &coordinates[a] - &coordinates[b])
                    } else {
                        (a, span)
                    };
                    let hits = bvh.intersect_all(
                        &(coordinates[from].clone(), along.clone()).into(),
                        surface_coordinates,
                        &elements,
                    );
                    let distances: Vec<Quantity<Length>> = dedupe(hits, margin)
                        .into_iter()
                        .filter(|&distance| distance < length - margin)
                        .collect();
                    if !distances.is_empty() {
                        let points: Vec<Coordinate<D>> = distances
                            .iter()
                            .map(|&distance| {
                                &coordinates[from] + &(&along * (distance.ratio(length)))
                            })
                            .collect();
                        let ordered = if from == a {
                            points
                        } else {
                            points.into_iter().rev().collect()
                        };
                        crossings.insert([a, b], ordered);
                    }
                    Ok(())
                }
                (inside, outside) if inside != outside => {
                    let hits = bvh.intersect_all(
                        &(coordinates[a].clone(), span.clone()).into(),
                        surface_coordinates,
                        &elements,
                    );
                    let distances: Vec<Quantity<Length>> = dedupe(hits, margin)
                        .into_iter()
                        .filter(|&distance| distance <= length + margin)
                        .collect();
                    if distances.is_empty() {
                        return Err("crossing missing on a sign-change edge");
                    }
                    if distances.len().is_multiple_of(2) {
                        return Err(
                            "edge crosses the tessellation an inconsistent number of times",
                        );
                    }
                    crossings.insert(
                        [a, b],
                        distances
                            .iter()
                            .map(|&distance| &coordinates[a] + &(&span * (distance.ratio(length))))
                            .collect(),
                    );
                    Ok(())
                }
                _ => {
                    let hits = bvh.intersect_all(
                        &(coordinates[a].clone(), span.clone()).into(),
                        surface_coordinates,
                        &elements,
                    );
                    let distances: Vec<Quantity<Length>> = dedupe(hits, margin)
                        .into_iter()
                        .filter(|&distance| distance <= length + margin)
                        .collect();
                    if distances.len() % 2 == 1 {
                        return Err(
                            "edge crosses the tessellation an inconsistent number of times",
                        );
                    }
                    if !distances.is_empty() {
                        crossings.insert(
                            [a, b],
                            distances
                                .iter()
                                .map(|&distance| {
                                    &coordinates[a] + &(&span * (distance.ratio(length)))
                                })
                                .collect(),
                        );
                    }
                    Ok(())
                }
            }
        })?;
        Ok((edges, signs, crossings))
    }
    pub fn tables(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        snapped: &HashSet<usize>,
    ) -> Result<Tables, &'static str> {
        let (_, signs, crossings) = self.edges_signs_crossings(mesh, classes, snapped)?;
        let coordinates = mesh.coordinates();
        let surface_coordinates = self.mesh().coordinates();
        let elements: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let contains = |point: &Coordinate<D>| {
            self.encloses(point, surface_coordinates, &elements, &normals, &directions)
        };
        let mut face_loops = HashMap::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    local_faces.iter().for_each(|face| {
                        let corners = from_fn::<_, 4, _>(|i| element[face[i]]);
                        let mut key = corners;
                        key.sort_unstable();
                        face_loops.entry(key).or_insert(corners);
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let mut segments = HashMap::new();
        face_loops.iter().try_for_each(|(key, corners)| {
            let cut = face_cut(corners, &signs, &crossings)?;
            let count = cut.endpoints.len();
            if count == 0 {
                return Ok(());
            }
            if count % 2 == 1 {
                return Err("refinement required at a face");
            }
            let target = if count == 2 {
                cut.sides[0]
            } else {
                let center = corners
                    .iter()
                    .map(|&node| coordinates[node].clone())
                    .sum::<Coordinate<D>>()
                    / 4.0;
                if contains(&center) {
                    Sign::Outside
                } else {
                    Sign::Inside
                }
            };
            let pairs: Vec<[Vertex; 2]> = (0..count)
                .filter(|&arc| cut.sides[arc] == target)
                .map(|arc| [cut.endpoints[arc], cut.endpoints[(arc + 1) % count]])
                .collect();
            if pairs.len() != count / 2 {
                return Err("inconsistent crossings around a face");
            }
            segments.insert(*key, pairs);
            Ok(())
        })?;
        Ok(Tables {
            signs,
            crossings,
            faces: face_loops,
            segments,
        })
    }
    pub(super) fn tables_generic(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        snapped: &HashSet<usize>,
    ) -> Result<GenericTables, &'static str> {
        let (_, signs, crossings) = self.edges_signs_crossings(mesh, classes, snapped)?;
        let coordinates = mesh.coordinates();
        let surface_coordinates = self.mesh().coordinates();
        let elements: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let contains = |point: &Coordinate<D>| {
            self.encloses(point, surface_coordinates, &elements, &normals, &directions)
        };
        let mut face_loops = HashMap::<Vec<usize>, Vec<usize>>::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let owners = face_owners(block);
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    oriented_element_faces(block, element, local, owners.as_deref())
                        .into_iter()
                        .for_each(|corners| {
                            let mut key = corners.clone();
                            key.sort_unstable();
                            face_loops.entry(key).or_insert(corners);
                        })
                }
            });
            offset += block.number_of_elements();
        });
        let mut segments = HashMap::new();
        face_loops.iter().try_for_each(|(key, corners)| {
            let cut = face_cut(corners, &signs, &crossings)?;
            let count = cut.endpoints.len();
            if count == 0 {
                return Ok(());
            }
            if count % 2 == 1 {
                return Err("refinement required at a face");
            }
            let n = corners.len();
            let target = if count == 2 {
                cut.sides[0]
            } else {
                let center = corners
                    .iter()
                    .map(|&node| coordinates[node].clone())
                    .sum::<Coordinate<D>>()
                    / n as Scalar;
                if contains(&center) {
                    Sign::Outside
                } else {
                    Sign::Inside
                }
            };
            let pairs: Vec<[Vertex; 2]> = (0..count)
                .filter(|&arc| cut.sides[arc] == target)
                .map(|arc| [cut.endpoints[arc], cut.endpoints[(arc + 1) % count]])
                .collect();
            if pairs.len() != count / 2 {
                return Err("inconsistent crossings around a face");
            }
            segments.insert(key.clone(), pairs);
            Ok(())
        })?;
        Ok(GenericTables {
            signs,
            crossings,
            faces: face_loops,
            segments,
        })
    }
}
