#[cfg(test)]
mod test;

use super::SolidOracle;
use crate::{
    geometry::{Coordinate, Direction},
    math::{Quantity, Tensor},
    units::Length,
};

/// Volumes in priority order: a point belongs to the first volume containing it.
pub struct Partition<O> {
    volumes: Vec<O>,
}

impl<O: SolidOracle> Partition<O> {
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
    ///
    /// The normal points out of the region, so it is reversed on the wall of an earlier volume.
    pub fn project(
        &self,
        point: &Coordinate<3>,
        index: usize,
    ) -> Option<(Coordinate<3>, Direction<3>)> {
        let (owner, closest, normal) = self.nearest(point, index)?;
        self.violator(&closest, index, &[owner])
            .is_none()
            .then(|| (closest, if owner < index { -normal } else { normal }))
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
        let (owner, mut closest, _) = self
            .nearest(point, index)
            .ok_or("partition has no volume to project onto")?;
        let Some(hiding) = self.violator(&closest, index, &[owner]) else {
            return Ok(closest);
        };
        let mut converged = false;
        for _ in 0..iterations {
            closest = self.surface(hiding, &closest)?;
            if self.volumes[owner].signed_distance(&closest).abs() <= tolerance {
                converged = true;
                break;
            }
            closest = self.surface(owner, &closest)?;
            if self.volumes[hiding].signed_distance(&closest).abs() <= tolerance {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("projections between the two surfaces did not converge");
        }
        if self.violator(&closest, index, &[owner, hiding]).is_some() {
            return Err("crease point lies inside a third volume");
        }
        Ok(closest)
    }
    /// A point of the region boundary on the segment between a point inside the region and one outside it.
    ///
    /// The segment is halved, keeping the end that is inside, until it is no longer than
    /// `tolerance`; the inside end is returned. This needs no surface structure, so it works
    /// where the projections do not, but the point found is not the nearest one.
    pub fn boundary_on_segment(
        &self,
        inside: &Coordinate<3>,
        outside: &Coordinate<3>,
        index: usize,
        tolerance: Quantity<Length>,
    ) -> Result<Coordinate<3>, &'static str> {
        let zero = Quantity::default();
        if tolerance <= zero {
            return Err("tolerance must be positive");
        }
        if self.signed_distance(inside, index) > zero {
            return Err("the first end of the segment is not inside the region");
        }
        if self.signed_distance(outside, index) <= zero {
            return Err("the second end of the segment is not outside the region");
        }
        let (mut inside, mut outside) = (inside.clone(), outside.clone());
        while (&outside - &inside).norm() > tolerance {
            let middle = (&inside + &outside) * 0.5;
            if self.signed_distance(&middle, index) <= zero {
                inside = middle;
            } else {
                outside = middle;
            }
        }
        Ok(inside)
    }
    fn surface(&self, volume: usize, point: &Coordinate<3>) -> Result<Coordinate<3>, &'static str> {
        self.volumes[volume]
            .project(point)
            .map(|(closest, _)| closest)
            .ok_or("a volume has no surface to project onto")
    }
    fn nearest(
        &self,
        point: &Coordinate<3>,
        index: usize,
    ) -> Option<(usize, Coordinate<3>, Direction<3>)> {
        self.volumes[..=index]
            .iter()
            .enumerate()
            .filter_map(|(owner, volume)| {
                volume
                    .project(point)
                    .map(|(closest, normal)| (owner, closest, normal))
            })
            .min_by(|(_, a, _), (_, b, _)| (a - point).norm().total_cmp(&(b - point).norm()))
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
