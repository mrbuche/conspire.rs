#[cfg(test)]
mod test;

use super::{
    Class, SNAP_HARD, SNAP_QUALITY, SNAP_SOFT,
    geometry::signed_volume,
    topology::{element_edges, element_faces},
};
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh,
            quality::metrics::{Kind, minimum_scaled_jacobian},
            tessellation::{D, Tessellation},
        },
    },
    math::{Scalar, Tensor},
};
use std::collections::{HashMap, HashSet};

impl Tessellation {
    pub(super) fn snap(
        &self,
        mesh: Mesh<D>,
        classes: &[Class],
    ) -> Result<(Mesh<D>, HashSet<usize>), &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut lengths = HashMap::<usize, Scalar>::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    element_edges(&element_faces(block, element))
                        .into_iter()
                        .for_each(|[a, b]| {
                            let length = (&coordinates[b] - &coordinates[a]).norm();
                            [a, b].into_iter().for_each(|node| {
                                lengths
                                    .entry(node)
                                    .and_modify(|shortest| *shortest = shortest.min(length))
                                    .or_insert(length);
                            })
                        })
                }
            });
            offset += block.number_of_elements();
        });
        let cells: Vec<Vec<usize>> = mesh
            .iter()
            .flat_map(|block| block.iter().map(|element| element.to_vec()))
            .collect();
        let incidents = mesh.node_element_connectivity().to_vec();
        let mut working = mesh.coordinates().clone();
        let mut snapped = HashSet::new();
        let mut candidates: Vec<usize> = lengths.keys().copied().collect();
        candidates.sort_unstable();
        candidates.into_iter().for_each(|node| {
            if let Some((closest, _)) =
                bvh.closest_point(&working[node], surface_coordinates, &elements)
            {
                let distance = (&closest - &working[node]).norm();
                let shortest = lengths[&node];
                let accept = if distance < SNAP_HARD * shortest {
                    true
                } else if distance < SNAP_SOFT * shortest {
                    let retained: Vec<&Vec<usize>> = incidents[node]
                        .iter()
                        .filter(|&&cell| classes[cell] == Class::Inside)
                        .map(|&cell| &cells[cell])
                        .collect();
                    let quality = |coordinates: &Coordinates<D>| {
                        retained
                            .iter()
                            .map(|cell| {
                                minimum_scaled_jacobian(Kind::Hexahedron, cell, coordinates)
                            })
                            .fold(Scalar::INFINITY, Scalar::min)
                    };
                    let before = quality(&working);
                    let previous = working[node].clone();
                    working[node] = closest.clone();
                    let keep = retained.is_empty() || quality(&working) >= before.min(SNAP_QUALITY);
                    if !keep {
                        working[node] = previous;
                    }
                    keep
                } else {
                    false
                };
                if accept {
                    working[node] = closest;
                    snapped.insert(node);
                }
            }
        });
        let (connectivities, _) = mesh.into();
        let hexes: Vec<[usize; 8]> = Vec::try_from(connectivities)?;
        Ok((
            (vec![Connectivity::Hexahedral(hexes.into())], working).into(),
            snapped,
        ))
    }
    /// The generic (arbitrary-polyhedra) analogue of `snap()`. The hex path's
    /// scaled-Jacobian quality gate doesn't generalize to arbitrary face
    /// counts, so this instead gates on relative signed volume: a candidate
    /// snap is accepted only if every retained incident cell keeps its
    /// volume-to-original ratio at or above the (softened) quality floor,
    /// mirroring the hex gate's relative-quality-drop structure.
    pub(super) fn snap_generic(
        &self,
        mesh: Mesh<D>,
        classes: &[Class],
    ) -> Result<(Mesh<D>, HashSet<usize>), &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut lengths = HashMap::<usize, Scalar>::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    element_edges(&element_faces(block, element))
                        .into_iter()
                        .for_each(|[a, b]| {
                            let length = (&coordinates[b] - &coordinates[a]).norm();
                            [a, b].into_iter().for_each(|node| {
                                lengths
                                    .entry(node)
                                    .and_modify(|shortest| *shortest = shortest.min(length))
                                    .or_insert(length);
                            })
                        })
                }
            });
            offset += block.number_of_elements();
        });
        let mut cells = Vec::<Vec<Vec<usize>>>::with_capacity(classes.len());
        let mut original_volumes = Vec::<Scalar>::with_capacity(classes.len());
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Inside {
                    let faces = element_faces(block, element);
                    original_volumes.push(signed_volume(&faces, coordinates));
                    cells.push(faces)
                } else {
                    original_volumes.push(1.0);
                    cells.push(Vec::new())
                }
            });
            offset += block.number_of_elements();
        });
        let mut incidents: Vec<Vec<usize>> = vec![Vec::new(); coordinates.len()];
        cells.iter().enumerate().for_each(|(cell, faces)| {
            faces
                .iter()
                .flatten()
                .for_each(|&node| incidents[node].push(cell))
        });
        incidents.iter_mut().for_each(|nodes| {
            nodes.sort_unstable();
            nodes.dedup();
        });
        let mut working = mesh.coordinates().clone();
        let mut snapped = HashSet::new();
        let mut candidates: Vec<usize> = lengths.keys().copied().collect();
        candidates.sort_unstable();
        candidates.into_iter().for_each(|node| {
            if let Some((closest, _)) =
                bvh.closest_point(&working[node], surface_coordinates, &elements)
            {
                let distance = (&closest - &working[node]).norm();
                let shortest = lengths[&node];
                let accept = if distance < SNAP_HARD * shortest {
                    true
                } else if distance < SNAP_SOFT * shortest {
                    let retained: Vec<usize> = incidents[node]
                        .iter()
                        .filter(|&&cell| classes[cell] == Class::Inside)
                        .copied()
                        .collect();
                    let quality = |coordinates: &Coordinates<D>| {
                        retained
                            .iter()
                            .map(|&cell| {
                                signed_volume(&cells[cell], coordinates) / original_volumes[cell]
                            })
                            .fold(Scalar::INFINITY, Scalar::min)
                    };
                    let before = quality(&working);
                    let previous = working[node].clone();
                    working[node] = closest.clone();
                    let keep = retained.is_empty() || quality(&working) >= before.min(SNAP_QUALITY);
                    if !keep {
                        working[node] = previous;
                    }
                    keep
                } else {
                    false
                };
                if accept {
                    working[node] = closest;
                    snapped.insert(node);
                }
            }
        });
        let (connectivities, _) = mesh.into();
        Ok(((connectivities.into_members(), working).into(), snapped))
    }
}
