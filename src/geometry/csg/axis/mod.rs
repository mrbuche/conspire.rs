//! The line that a solid of revolution turns about, and positions measured from it.

use crate::{
    geometry::{Coordinate, Direction},
    math::{Quantity, Tensor},
    units::Length,
};
use core::cmp::Ordering::Greater;

/// A line through `origin` along a unit `direction`.
pub(super) struct Axis {
    origin: Coordinate<3>,
    direction: Direction<3>,
}

impl Axis {
    /// The line through `origin` along `direction`, which is made a unit vector and cannot be zero.
    pub fn new(origin: Coordinate<3>, direction: Direction<3>) -> Result<Self, &'static str> {
        let norm = direction.norm();
        if norm.partial_cmp(&0.0) != Some(Greater) || !norm.value().is_finite() {
            return Err("the axis has no direction");
        }
        Ok(Self {
            origin,
            direction: direction.normalized(),
        })
    }
    pub fn direction(&self) -> &Direction<3> {
        &self.direction
    }
    /// How far along the axis from the origin a point is.
    pub fn along(&self, point: &Coordinate<3>) -> Quantity<Length> {
        &(point - &self.origin) * &self.direction
    }
    /// The offset of a point from the axis, which is across it, and the length of that offset.
    pub fn across(&self, point: &Coordinate<3>) -> (Coordinate<3>, Quantity<Length>) {
        let offset = point - &self.origin;
        let across = &offset - &(&self.direction * (&offset * &self.direction));
        let distance = across.norm();
        (across, distance)
    }
    /// The unit direction of an offset from the axis, or any direction across the axis where there
    /// is no offset, since every one is as close.
    pub fn outward(&self, across: &Coordinate<3>, distance: Quantity<Length>) -> Direction<3> {
        if distance > Quantity::default() {
            across.normalized()
        } else {
            self.perpendicular()
        }
    }
    /// The point `along` the axis and `distance` from it in the unit direction `outward`.
    pub fn at(
        &self,
        along: Quantity<Length>,
        outward: &Direction<3>,
        distance: Quantity<Length>,
    ) -> Coordinate<3> {
        &(&self.origin + &(&self.direction * along)) + &(outward * distance)
    }
    fn perpendicular(&self) -> Direction<3> {
        let axis = &self.direction;
        let seed = if axis[0].abs() < 0.9 {
            Direction::const_from([1.0, 0.0, 0.0])
        } else {
            Direction::const_from([0.0, 1.0, 0.0])
        };
        (&seed - &(axis * (&seed * axis))).normalized()
    }
}
