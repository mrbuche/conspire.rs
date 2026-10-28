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
        let (owner, closest) = self.nearest(point, index)?;
        self.violator(&closest, index, &[owner])
            .is_none()
            .then_some(closest)
    }
    /// A point of the region boundary near the query, including where it lies on a crease.
    ///
    /// Where [`Self::project`] is exact it is used. Otherwise the nearest candidate is hidden
    /// inside another volume, and projections alternate between the surface the candidate lies
    /// on and the surface hiding it until the iterate is within `tolerance` of both. The
    /// result lies on the crease between them, which is not necessarily the nearest such point.
    pub fn project_to_crease(
        &self,
        point: &Coordinate<3>,
        index: usize,
        tolerance: Quantity<Length>,
        iterations: usize,
    ) -> Result<Coordinate<3>, &'static str> {
        let (owner, mut closest) = self
            .nearest(point, index)
            .ok_or("partition has no volume to project onto")?;
        let Some(hiding) = self.violator(&closest, index, &[owner]) else {
            return Ok(closest);
        };
        let converged = (0..iterations).any(|_| {
            closest = self.volumes[hiding].project(&closest);
            if self.volumes[owner].signed_distance(&closest).abs() <= tolerance {
                return true;
            }
            closest = self.volumes[owner].project(&closest);
            self.volumes[hiding].signed_distance(&closest).abs() <= tolerance
        });
        if !converged {
            return Err("projections between the two surfaces did not converge");
        }
        if self.violator(&closest, index, &[owner, hiding]).is_some() {
            return Err("crease point lies inside a third volume");
        }
        Ok(closest)
    }
    fn nearest(&self, point: &Coordinate<3>, index: usize) -> Option<(usize, Coordinate<3>)> {
        self.volumes[..=index]
            .iter()
            .enumerate()
            .map(|(owner, volume)| (owner, volume.project(point)))
            .min_by(|(_, a), (_, b)| (a - point).norm().total_cmp(&(b - point).norm()))
    }
    fn violator(&self, point: &Coordinate<3>, index: usize, skip: &[usize]) -> Option<usize> {
        let zero = Quantity::default();
        (0..=index)
            .filter(|other| !skip.contains(other))
            .find(|&other| {
                let distance = self.volumes[other].signed_distance(point);
                if other == index {
                    distance > zero
                } else {
                    distance < zero
                }
            })
    }
}
