#[cfg(test)]
mod test;

use super::{D, Tessellation, cut::DIRECTIONS};
use crate::{
    geometry::{Coordinate, oracle::Oracle},
    math::{Quantity, Tensor},
    units::Length,
};

impl Oracle for Tessellation {
    fn project(&self, point: &Coordinate<D>) -> Coordinate<D> {
        let mesh = self.mesh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        self.bvh()
            .closest_point(point, mesh.coordinates(), &elements)
            .map_or_else(|| point.clone(), |(closest, _)| closest)
    }
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let mesh = self.mesh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let normals = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let distance = (&self.project(point) - point).norm();
        if self.encloses(point, mesh.coordinates(), &elements, &normals, &directions) {
            -distance
        } else {
            distance
        }
    }
}
