#[cfg(test)]
mod test;

use super::Class;
use crate::{
    geometry::{Coordinate, Coordinates, bvh::Hit, mesh::Mesh, mesh::tessellation::D},
    math::{CrossProduct, Scalar, Tensor},
};
use std::{array::from_fn, collections::HashMap};

pub(super) fn dedupe(hits: Vec<Hit>, margin: Scalar) -> Vec<Scalar> {
    let mut distances = Vec::new();
    hits.iter().for_each(|hit| {
        if distances
            .last()
            .is_none_or(|&last| hit.distance() - last > margin)
        {
            distances.push(hit.distance());
        }
    });
    distances
}

pub(super) fn face_area(face: &[usize], coordinates: &Coordinates<D>) -> Scalar {
    let middle = face
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / face.len() as Scalar;
    (0..face.len())
        .map(|i| {
            let one = &coordinates[face[i]] - &middle;
            let two = &coordinates[face[(i + 1) % face.len()]] - &middle;
            one.cross(&two).norm() / 2.0
        })
        .sum()
}

pub(super) fn star_volume(faces: &[Vec<usize>], coordinates: &Coordinates<D>) -> Scalar {
    let nodes: std::collections::HashSet<usize> = faces.iter().flatten().copied().collect();
    let centroid = nodes
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / nodes.len() as Scalar;
    faces
        .iter()
        .map(|face| {
            let middle = &(face
                .iter()
                .map(|&node| coordinates[node].clone())
                .sum::<Coordinate<D>>()
                / face.len() as Scalar)
                - &centroid;
            (0..face.len())
                .map(|i| {
                    let one = &coordinates[face[i]] - &centroid;
                    let two = &coordinates[face[(i + 1) % face.len()]] - &centroid;
                    (one.cross(&two) * &middle).abs() / 6.0
                })
                .sum::<Scalar>()
        })
        .sum()
}

pub(super) fn signed_volume(faces: &[Vec<usize>], coordinates: &Coordinates<D>) -> Scalar {
    let nodes: std::collections::HashSet<usize> = faces.iter().flatten().copied().collect();
    let origin = nodes
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / nodes.len() as Scalar;
    faces
        .iter()
        .map(|face| {
            let middle = &(face
                .iter()
                .map(|&node| coordinates[node].clone())
                .sum::<Coordinate<D>>()
                / face.len() as Scalar)
                - &origin;
            (0..face.len())
                .map(|i| {
                    let one = &coordinates[face[i]] - &origin;
                    let two = &coordinates[face[(i + 1) % face.len()]] - &origin;
                    (one.cross(&two) * &middle) / 6.0
                })
                .sum::<Scalar>()
        })
        .sum()
}

pub(super) fn contained(mesh: &Mesh<D>, classes: &[Class]) -> bool {
    let mut faces = HashMap::<[usize; 4], (usize, u8)>::new();
    let mut offset = 0;
    mesh.iter().for_each(|block| {
        let local_faces = block.local_faces();
        block.iter().enumerate().for_each(|(local, element)| {
            local_faces.iter().for_each(|face| {
                let mut key = from_fn(|i| element[face[i]]);
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
