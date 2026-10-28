#[cfg(test)]
mod test;

use super::Oracle;
use crate::{geometry::Coordinate, math::Quantity, units::Length};

/// Volumes in priority order: a point belongs to the first volume containing it.
pub struct Partition<O> {
    volumes: Vec<O>,
}

impl<O: Oracle> Partition<O> {
    pub fn new(volumes: Vec<O>) -> Self {
        Self { volumes }
    }
    pub fn number_of_volumes(&self) -> usize {
        self.volumes.len()
    }
    /// The volume owning the point, or `None` where the point is inside no volume.
    pub fn owner(&self, point: &Coordinate<3>) -> Option<usize> {
        self.volumes
            .iter()
            .position(|volume| volume.signed_distance(point) <= Quantity::default())
    }
    /// Signed distance to the region of a volume left after removing every earlier volume.
    pub fn signed_distance(&self, point: &Coordinate<3>, index: usize) -> Quantity<Length> {
        self.volumes[..index].iter().fold(
            self.volumes[index].signed_distance(point),
            |distance, earlier| distance.max(-earlier.signed_distance(point)),
        )
    }
}
