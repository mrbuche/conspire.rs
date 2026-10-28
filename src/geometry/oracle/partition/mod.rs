#[cfg(test)]
mod test;

use super::Oracle;
use crate::{
    geometry::Coordinate,
    math::{Quantity, Tensor},
    units::Length,
};

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
    /// Closest point on the boundary of the region of a volume left after removing every earlier volume.
    ///
    /// Each volume offers its own closest boundary point. A candidate lies on the boundary of
    /// the region only if it is inside the volume and strictly inside no other earlier volume.
    /// The nearest candidate is the exact answer when it passes; otherwise the closest point is
    /// on a crease between two surfaces and `None` is returned.
    pub fn project(&self, point: &Coordinate<3>, index: usize) -> Option<Coordinate<3>> {
        let (owner, closest) = self.volumes[..=index]
            .iter()
            .enumerate()
            .map(|(owner, volume)| (owner, volume.project(point)))
            .min_by(|(_, a), (_, b)| (a - point).norm().total_cmp(&(b - point).norm()))?;
        let zero = Quantity::default();
        (0..=index)
            .filter(|&other| other != owner)
            .all(|other| {
                let distance = self.volumes[other].signed_distance(&closest);
                if other == index {
                    distance <= zero
                } else {
                    distance >= zero
                }
            })
            .then_some(closest)
    }
}
