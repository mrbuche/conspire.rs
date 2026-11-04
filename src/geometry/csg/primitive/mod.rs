//! A closed enum over the analytic primitives, so a single [`Union`](super::ops::Union) or
//! [`Partition`](crate::geometry::solid::Partition) can mix shapes without trait objects.

#[cfg(test)]
mod test;

use super::{Cuboid, Sphere};
use crate::{
    geometry::{
        Coordinate,
        solid::{Projection, SolidOracle},
    },
    math::Quantity,
    units::Length,
};

/// Any one analytic primitive.
pub enum Primitive {
    Cuboid(Cuboid),
    Sphere(Sphere),
}

impl From<Cuboid> for Primitive {
    fn from(primitive: Cuboid) -> Self {
        Self::Cuboid(primitive)
    }
}

impl From<Sphere> for Primitive {
    fn from(primitive: Sphere) -> Self {
        Self::Sphere(primitive)
    }
}

impl SolidOracle for Primitive {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        match self {
            Self::Cuboid(primitive) => primitive.signed_distance(point),
            Self::Sphere(primitive) => primitive.signed_distance(point),
        }
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        match self {
            Self::Cuboid(primitive) => primitive.project(point),
            Self::Sphere(primitive) => primitive.project(point),
        }
    }
}
