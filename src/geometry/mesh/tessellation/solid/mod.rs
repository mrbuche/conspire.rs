#[cfg(test)]
mod test;

use super::{D, NORMALIZED_DIRECTIONS, Tessellation};
use crate::{
    geometry::{Coordinate, Direction, DirectionsRef, solid::SolidOracle},
    math::{Quantity, Tensor},
    units::Length,
};

impl SolidOracle for Tessellation {
    fn project(&self, point: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let mesh = self.mesh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, _> = self.normals().iter().flatten().collect();
        self.bvh()
            .closest_point(point, mesh.coordinates(), &elements)
            .map(|(closest, face)| (closest, normals[face].normalized()))
    }
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let mesh = self.mesh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let normals = self.normals().iter().flatten().collect();
        let directions = NORMALIZED_DIRECTIONS;
        let distance = self
            .project(point)
            .map_or(Quantity::new(f64::INFINITY), |(closest, _)| {
                (&closest - point).norm()
            });
        if self.encloses(point, mesh.coordinates(), &elements, &normals, &directions) {
            -distance
        } else {
            distance
        }
    }
}
