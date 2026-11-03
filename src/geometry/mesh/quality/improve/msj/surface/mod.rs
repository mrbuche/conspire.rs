#[cfg(test)]
mod test;

use super::{CAP, TOLERANCE, chunk_size};
use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef, Direction, DirectionsRef, bbox::BoundingBox,
        mesh::Tessellation,
    },
    math::{Quantity, QuantityVector, Scalar, Tensor},
    units::{Dimensionless, Length},
};
use std::{array::from_fn, thread::scope};

const CREASE: Quantity = Dimensionless::of(0.866);

const RADIUS: Quantity = Dimensionless::of(0.2);

const TIE: Scalar = 1.0e-6;

pub(super) struct Plane {
    pub(super) boundary: usize,
    pub(super) normal: Direction<3>,
    pub(super) offset: Quantity<Length>,
    pub(super) slot: usize,
}

impl Plane {
    pub(super) fn residual(&self, position: &Coordinate<3>) -> Quantity<Length> {
        position * &self.normal - self.offset
    }
}

pub(super) struct Surface<'a> {
    normals: DirectionsRef<'a, 3>,
    surface: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    target: &'a Tessellation,
}

impl<'a> Surface<'a> {
    pub(super) fn new(target: &'a Tessellation) -> Self {
        let mesh = target.mesh();
        Self {
            normals: target.normals().iter().flatten().collect(),
            surface: mesh.coordinates(),
            elements: mesh.connectivities().iter().flatten().collect(),
            target,
        }
    }
    pub(super) fn planes(
        &self,
        boundary: &[usize],
        sizes: &QuantityVector<Length>,
        coordinates: &Coordinates<3>,
    ) -> Vec<Plane> {
        let bvh = self.target.bvh();
        let (surface, elements, normals) = (self.surface, &self.elements, &self.normals);
        let chunk = chunk_size(boundary.len());
        scope(|scope| {
            boundary
                .chunks(chunk)
                .enumerate()
                .map(|(block, nodes)| {
                    scope.spawn(move || {
                        let mut planes = Vec::new();
                        nodes.iter().enumerate().for_each(|(offset, &node)| {
                            let index = block * chunk + offset;
                            let position = &coordinates[node];
                            let ties =
                                bvh.closest_points(position, surface, elements, sizes[node] * TIE);
                            let Some((point, _)) = ties.first().cloned() else {
                                return;
                            };
                            let tied: Vec<usize> =
                                ties.iter().map(|&(_, triangle)| triangle).collect();
                            let reach = sizes[node] * RADIUS;
                            let (low, high) = (
                                from_fn(|i| point[i] - reach).into(),
                                from_fn(|i| point[i] + reach).into(),
                            );
                            let query: BoundingBox<3> = [&low, &high]
                                .into_iter()
                                .collect::<CoordinatesRef<'_, 3>>()
                                .into();
                            let mut kept = Vec::<Direction<3>>::new();
                            for triangle in tied.iter().copied().chain(bvh.overlapping(&query)) {
                                if kept.len() == CAP {
                                    break;
                                }
                                let nodes = elements[triangle];
                                let tie = tied.contains(&triangle);
                                if !tie
                                    && !query.overlaps_triangle(
                                        &surface[nodes[0]],
                                        &surface[nodes[1]],
                                        &surface[nodes[2]],
                                    )
                                {
                                    continue;
                                }
                                let normal = normals[triangle].clone();
                                if kept.iter().any(|other| (other * &normal).abs() > CREASE) {
                                    continue;
                                }
                                let plane = Plane {
                                    boundary: index,
                                    offset: &surface[elements[triangle][0]] * &normal,
                                    normal,
                                    slot: kept.len(),
                                };
                                if !tie && plane.residual(position).abs() > sizes[node] * TOLERANCE
                                {
                                    continue;
                                }
                                kept.push(plane.normal.clone());
                                planes.push(plane)
                            }
                        });
                        planes
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        })
    }
}
