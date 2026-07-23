#[cfg(test)]
mod test;

use super::{Class, EDGES, SNAP_HARD, SNAP_QUALITY, SNAP_SOFT};
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
                    EDGES.iter().for_each(|&[a, b]| {
                        let length = (&coordinates[element[b]] - &coordinates[element[a]]).norm();
                        [element[a], element[b]].into_iter().for_each(|node| {
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
}
