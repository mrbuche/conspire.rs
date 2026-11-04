#[cfg(test)]
mod test;

use super::axis::Axis;
use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Projection, SolidOracle},
    },
    math::{Quantity, Tensor},
    units::Length,
};

/// A ring torus: a tube of `minor_radius` swept around an axis at the `major_radius`.
pub struct Torus {
    axis: Axis,
    major_radius: Quantity<Length>,
    minor_radius: Quantity<Length>,
}

impl Torus {
    /// The torus about `axis` through `center`. The minor radius must be positive and less than
    /// the major radius, so the tube does not cross itself, and the axis cannot be zero.
    pub fn new(
        center: Coordinate<3>,
        axis: Direction<3>,
        major_radius: Quantity<Length>,
        minor_radius: Quantity<Length>,
    ) -> Result<Self, &'static str> {
        if minor_radius <= Quantity::default() || major_radius <= minor_radius {
            return Err("torus needs 0 < minor_radius < major_radius");
        }
        Ok(Self {
            axis: Axis::new(center, axis)?,
            major_radius,
            minor_radius,
        })
    }
    /// The point of the circle through the middle of the tube nearest the query, which is the
    /// point of the circle in the direction of the query from the axis, and the offset from it to
    /// the query with its length.
    fn offset(
        &self,
        point: &Coordinate<3>,
    ) -> (Coordinate<3>, Direction<3>, Coordinate<3>, Quantity<Length>) {
        let (across, distance) = self.axis.across(point);
        let outward = self.axis.outward(&across, distance);
        let ring = self
            .axis
            .at(Quantity::default(), &outward, self.major_radius);
        let offset = point - &ring;
        let length = offset.norm();
        (ring, outward, offset, length)
    }
}

impl SolidOracle for Torus {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        self.offset(point).3 - self.minor_radius
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let (ring, outward, offset, length) = self.offset(point);
        let normal = if length > Quantity::default() {
            offset.normalized()
        } else {
            outward
        };
        Some(Projection {
            point: &ring + &normal * self.minor_radius,
            normal,
            exact: true,
        })
    }
}
