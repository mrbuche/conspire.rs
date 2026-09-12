#[cfg(test)]
mod test;

use super::{Class, DIRECTIONS, RegionClass, topology::element_faces};
use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef, Direction, DirectionsRef,
        bbox::BoundingBox,
        mesh::{Mesh, tessellation::D, tessellation::Tessellation},
    },
    math::Tensor,
};
use std::collections::{HashMap, hash_map::Entry};

/// Classifies a background mesh against an ordered slice of surfaces.
///
/// The multi-surface counterpart of [`Tessellation::classify`]. A cell is
/// [`Cut`](RegionClass::Cut) by every surface whose triangles overlap its
/// bounding box; an uncut cell is flood-filled, together with its uncut
/// neighbors, to the [`Inside`](RegionClass::Inside) of the first surface
/// (in `surfaces` order) that encloses it, or [`Outside`](RegionClass::Outside)
/// if none do. `surfaces` should therefore be given in containment/priority
/// order (innermost/most specific first): a cell inside a nested surface is
/// also inside its container, and the nested surface should win.
///
/// A cell `Cut` by more than one surface straddles more than one material
/// boundary at once; resolving such a cell's geometry into per-region
/// polyhedra is not yet implemented.
pub fn classify_regions(surfaces: &[Tessellation], mesh: &Mesh<D>) -> Vec<RegionClass> {
    struct Surface<'a> {
        coordinates: &'a Coordinates<D>,
        elements: Vec<&'a [usize]>,
        normals: DirectionsRef<'a, D>,
        tessellation: &'a Tessellation,
    }
    let directions = DIRECTIONS.map(|direction| direction.normalized());
    let data: Vec<Surface> = surfaces
        .iter()
        .map(|tessellation| {
            let surface = tessellation.mesh();
            Surface {
                coordinates: surface.coordinates(),
                elements: surface.connectivities().iter().flatten().collect(),
                normals: tessellation.normals().iter().flatten().collect(),
                tessellation,
            }
        })
        .collect();
    let coordinates = mesh.coordinates();
    let number_of_elements = mesh.number_of_elements();
    let mut cut_by: Vec<Vec<usize>> = vec![Vec::new(); number_of_elements];
    mesh.iter()
        .flat_map(|block| {
            block
                .iter()
                .map(move |element| block.element_nodes(element))
        })
        .zip(cut_by.iter_mut())
        .for_each(|(nodes, hits)| {
            let bbox: BoundingBox<D> = nodes
                .iter()
                .map(|&node| &coordinates[node])
                .collect::<CoordinatesRef<'_, D>>()
                .into();
            data.iter()
                .enumerate()
                .for_each(|(surface_index, surface)| {
                    let hit = surface
                        .tessellation
                        .bvh()
                        .overlapping(&bbox)
                        .into_iter()
                        .any(|triangle| {
                            let nodes = surface.elements[triangle];
                            bbox.overlaps_triangle(
                                &surface.coordinates[nodes[0]],
                                &surface.coordinates[nodes[1]],
                                &surface.coordinates[nodes[2]],
                            )
                        });
                    if hit {
                        hits.push(surface_index);
                    }
                });
        });
    let mut faces = HashMap::new();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); number_of_elements];
    let mut offset = 0;
    mesh.iter().for_each(|block| {
        block.iter().enumerate().for_each(|(local, element)| {
            let index = offset + local;
            if cut_by[index].is_empty() {
                element_faces(block, element).into_iter().for_each(|face| {
                    let mut key = face;
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
    let mut classes: Vec<RegionClass> = cut_by
        .iter()
        .map(|hits| {
            if hits.is_empty() {
                RegionClass::Outside
            } else {
                RegionClass::Cut(hits.clone())
            }
        })
        .collect();
    let mut visited: Vec<bool> = cut_by.iter().map(|hits| !hits.is_empty()).collect();
    let mut stack = Vec::new();
    (0..number_of_elements).for_each(|seed| {
        if !visited[seed] {
            let class = data
                .iter()
                .enumerate()
                .find_map(|(surface_index, surface)| {
                    surface
                        .tessellation
                        .encloses(
                            &centroids[seed],
                            surface.coordinates,
                            &surface.elements,
                            &surface.normals,
                            &directions,
                        )
                        .then_some(RegionClass::Inside(surface_index))
                })
                .unwrap_or(RegionClass::Outside);
            visited[seed] = true;
            stack.push(seed);
            while let Some(index) = stack.pop() {
                classes[index] = class.clone();
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

impl Tessellation {
    pub fn classify(&self, mesh: &Mesh<D>) -> Vec<Class> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let number_of_elements = mesh.number_of_elements();
        let mut cut = vec![false; number_of_elements];
        mesh.iter()
            .flat_map(|block| {
                block
                    .iter()
                    .map(move |element| block.element_nodes(element))
            })
            .zip(cut.iter_mut())
            .for_each(|(nodes, flag)| {
                let bbox: BoundingBox<D> = nodes
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
            block.iter().enumerate().for_each(|(local, element)| {
                let index = offset + local;
                if !cut[index] {
                    element_faces(block, element).into_iter().for_each(|face| {
                        let mut key = face;
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
    pub(super) fn encloses(
        &self,
        point: &Coordinate<D>,
        surface_coordinates: &Coordinates<D>,
        elements: &[&[usize]],
        normals: &DirectionsRef<'_, D>,
        directions: &[Direction<D>; 3],
    ) -> bool {
        directions
            .iter()
            .find_map(|direction| {
                let ray = (point.clone(), direction.clone()).into();
                match self.bvh().intersect(&ray, surface_coordinates, elements) {
                    None => Some(false),
                    Some(hit) => {
                        let normal = &normals[hit.index()];
                        let cosine = (direction * normal) / normal.norm();
                        (cosine.abs() > super::GRAZING_TOLERANCE).then_some(cosine > 0.0)
                    }
                }
            })
            .unwrap_or(false)
    }
}
