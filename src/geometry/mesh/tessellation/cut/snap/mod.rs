#[cfg(test)]
mod test;

use super::{
    Class, SNAP_FEATURE, SNAP_HARD, SNAP_QUALITY, SNAP_SOFT,
    geometry::signed_volume,
    topology::{element_edges, element_faces},
};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            quality::metrics::{Kind, minimum_scaled_jacobian},
            tessellation::{D, Tessellation, features::FeatureIndex},
        },
    },
    math::{Scalar, Tensor},
};
use std::collections::{HashMap, HashSet};

/// Which corner each node should take, when two would otherwise take the same
/// one and leave a cell degenerate. The nearer node wins.
fn claims(
    index: &FeatureIndex<'_>,
    coordinates: &Coordinates<D>,
    candidates: &[usize],
    radius: impl Fn(usize) -> Scalar,
) -> HashMap<usize, Coordinate<D>> {
    let mut nearest = HashMap::<usize, (usize, Scalar)>::new();
    candidates.iter().for_each(|&node| {
        if let Some((corner, distance)) = index.nearest_corner(&coordinates[node], radius(node))
            && nearest
                .get(&corner)
                .is_none_or(|&(_, held)| distance < held)
        {
            nearest.insert(corner, (node, distance));
        }
    });
    nearest
        .into_iter()
        .map(|(corner, (node, _))| (node, index.corner(corner).clone()))
        .collect()
}

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
        let radius = |node: usize| SNAP_FEATURE * lengths[&node];
        let widest = candidates
            .iter()
            .map(|&node| radius(node))
            .fold(0.0, Scalar::max);
        let index = self.features().index(widest);
        let corners = claims(&index, &working, &candidates, radius);
        candidates.into_iter().for_each(|node| {
            if let Some((closest, _)) =
                bvh.closest_point(&working[node], surface_coordinates, &elements)
            {
                let corner = corners.get(&node).cloned();
                let onto_corner = corner.is_some();
                let target = corner.unwrap_or(closest);
                let distance = (&target - &working[node]).norm();
                let shortest = lengths[&node];
                let limit = if onto_corner { SNAP_FEATURE } else { SNAP_SOFT };
                let accept = if distance >= limit * shortest {
                    false
                } else if distance < SNAP_HARD * shortest {
                    true
                } else {
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
                    working[node] = target.clone();
                    let keep = retained.is_empty() || quality(&working) >= before.min(SNAP_QUALITY);
                    if !keep {
                        working[node] = previous;
                    }
                    keep
                };
                if accept {
                    working[node] = target;
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
        let radius = |node: usize| SNAP_FEATURE * lengths[&node];
        let widest = candidates
            .iter()
            .map(|&node| radius(node))
            .fold(0.0, Scalar::max);
        let index = self.features().index(widest);
        let corners = claims(&index, &working, &candidates, radius);
        candidates.into_iter().for_each(|node| {
            if let Some((closest, _)) =
                bvh.closest_point(&working[node], surface_coordinates, &elements)
            {
                let corner = corners.get(&node).cloned();
                let onto_corner = corner.is_some();
                let target = corner.unwrap_or(closest);
                let distance = (&target - &working[node]).norm();
                let shortest = lengths[&node];
                let limit = if onto_corner { SNAP_FEATURE } else { SNAP_SOFT };
                let accept = if distance >= limit * shortest {
                    false
                } else if distance < SNAP_HARD * shortest {
                    true
                } else {
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
                    working[node] = target.clone();
                    let keep = retained.is_empty() || quality(&working) >= before.min(SNAP_QUALITY);
                    if !keep {
                        working[node] = previous;
                    }
                    keep
                };
                if accept {
                    working[node] = target;
                    snapped.insert(node);
                }
            }
        });
        let (connectivities, _) = mesh.into();
        Ok(((connectivities.into_members(), working).into(), snapped))
    }
}
