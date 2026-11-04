#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Projection, SolidOracle},
    },
    math::{Quantity, Tensor},
    units::Length,
};

/// A ball.
pub struct Sphere {
    center: Coordinate<3>,
    radius: Quantity<Length>,
}

impl Sphere {
    /// The ball of `radius` about `center`; `radius` must be positive.
    pub fn new(center: Coordinate<3>, radius: Quantity<Length>) -> Result<Self, &'static str> {
        if radius <= Quantity::default() {
            return Err("sphere radius must be positive");
        }
        Ok(Self { center, radius })
    }
}

impl SolidOracle for Sphere {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        (point - &self.center).norm() - self.radius
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let offset = point - &self.center;
        let normal = if offset.norm() > Quantity::default() {
            offset.normalized()
        } else {
            Direction::const_from([1.0, 0.0, 0.0])
        };
        Some(Projection {
            point: &self.center + &(&normal * self.radius),
            normal,
            exact: true,
        })
    }
}
