#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        mesh::{Connectivity, Mesh},
    },
    math::{CrossProduct, FxHashMap, FxHashSet, Scalar, Tensor, TensorVec},
};
use std::array::from_fn;

const ASCENT_ITERATIONS: usize = 64;
const PASSES: usize = 64;
const TOLERANCE: Scalar = 1.0e-6;

impl Mesh<3> {
    pub(super) fn restrict(&mut self) -> Result<(), &'static str> {
        for _ in 0..PASSES {
            let [Connectivity::Hexahedral(block)] = self.connectivities() else {
                return Err("restrict requires a single hexahedral block");
            };
            let local_faces = self.connectivities()[0].local_faces();
            let hexes: Vec<[usize; 8]> = block.iter().copied().collect();
            let coordinates = self.coordinates();
            let mut faces = FxHashMap::<[usize; 4], Vec<(usize, [usize; 4])>>::default();
            hexes.iter().enumerate().for_each(|(hex_index, hex)| {
                local_faces.iter().for_each(|local| {
                    let oriented: [usize; 4] = from_fn(|i| hex[local[i]]);
                    let mut key = oriented;
                    key.sort_unstable();
                    faces.entry(key).or_default().push((hex_index, oriented));
                })
            });
            let boundary: Vec<(usize, [usize; 4])> = faces
                .into_values()
                .filter_map(|mut group| (group.len() == 1).then(|| group.pop().unwrap()))
                .collect();
            let normals: Coordinates<3> = boundary
                .iter()
                .map(|(_, face)| {
                    let diagonal_0 = &coordinates[face[2]] - &coordinates[face[0]];
                    let diagonal_1 = &coordinates[face[3]] - &coordinates[face[1]];
                    diagonal_0.cross(diagonal_1).normalized()
                })
                .collect();
            let mut vertex_faces = FxHashMap::<usize, Vec<usize>>::default();
            boundary.iter().enumerate().for_each(|(index, (_, face))| {
                face.iter()
                    .for_each(|&node| vertex_faces.entry(node).or_default().push(index));
            });
            let mut face_counts = FxHashMap::default();
            boundary.iter().for_each(|&(hex_index, _)| {
                face_counts
                    .entry(hex_index)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            });
            let mut offenders = FxHashSet::default();
            vertex_faces.values().for_each(|indices| {
                if indices.len() > 1 {
                    let locals = indices.iter().map(|&i| &normals[i]).collect();
                    if !feasible(&locals) {
                        let worst = indices
                            .iter()
                            .copied()
                            .max_by_key(|&i| face_counts[&boundary[i].0])
                            .unwrap();
                        offenders.insert(boundary[worst].0);
                    }
                }
            });
            if offenders.is_empty() {
                return Ok(());
            }
            let mut remap = vec![usize::MAX; coordinates.len()];
            let mut new_coordinates = Coordinates::new();
            let mut connectivity = Vec::new();
            hexes
                .iter()
                .enumerate()
                .filter(|(index, _)| !offenders.contains(index))
                .for_each(|(_, hex)| {
                    connectivity.push(from_fn(|i| {
                        let node = hex[i];
                        if remap[node] == usize::MAX {
                            remap[node] = new_coordinates.len();
                            new_coordinates.push(coordinates[node].clone());
                        }
                        remap[node]
                    }))
                });
            *self = (
                vec![Connectivity::Hexahedral(connectivity.into())],
                new_coordinates,
            )
                .into();
        }
        Err("clearance restriction did not converge")
    }
}

fn feasible(normals: &CoordinatesRef<3>) -> bool {
    let mut e: Coordinate<3> = normals.iter().copied().sum();
    if e.norm() < TOLERANCE {
        e = normals[0].clone();
    }
    e = e.normalized();
    let mut best = Scalar::NEG_INFINITY;
    for _ in 0..ASCENT_ITERATIONS {
        let (worst_index, worst) = normals.iter().enumerate().map(|(i, &n)| (i, n * &e)).fold(
            (0, Scalar::INFINITY),
            |(bi, bv), (i, v)| {
                if v < bv { (i, v) } else { (bi, bv) }
            },
        );
        best = best.max(worst);
        let candidate = &e + &normals[worst_index];
        if candidate.norm() < TOLERANCE {
            break;
        }
        e = candidate.normalized();
    }
    best > TOLERANCE
}
