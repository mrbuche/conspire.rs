//! A closed enum over the leaf primitives, so a single [`Union`](super::ops::Union)
//! (or any combinator) can mix shapes without trait objects.

#[cfg(test)]
mod test;

use super::{Cuboid, CuboidOracle, Cylinder, CylinderOracle, Sphere, SphereOracle};
use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};

const D: usize = 3;

/// Any one analytic primitive.
pub enum Primitive {
    Cuboid(Cuboid),
    Sphere(Sphere),
    Cylinder(Cylinder),
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

impl From<Cylinder> for Primitive {
    fn from(primitive: Cylinder) -> Self {
        Self::Cylinder(primitive)
    }
}

impl Solid for Primitive {
    type Oracle = PrimitiveOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        match self {
            Self::Cuboid(primitive) => primitive.bounding_box(),
            Self::Sphere(primitive) => primitive.bounding_box(),
            Self::Cylinder(primitive) => primitive.bounding_box(),
        }
    }

    fn oracle(&self) -> Result<PrimitiveOracle, &'static str> {
        Ok(match self {
            Self::Cuboid(primitive) => PrimitiveOracle::Cuboid(primitive.oracle()?),
            Self::Sphere(primitive) => PrimitiveOracle::Sphere(primitive.oracle()?),
            Self::Cylinder(primitive) => PrimitiveOracle::Cylinder(primitive.oracle()?),
        })
    }
}

/// [`SolidOracle`] for a [`Primitive`].
pub enum PrimitiveOracle {
    Cuboid(CuboidOracle),
    Sphere(SphereOracle),
    Cylinder(CylinderOracle),
}

impl SolidOracle for PrimitiveOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        match self {
            Self::Cuboid(oracle) => oracle.project(query),
            Self::Sphere(oracle) => oracle.project(query),
            Self::Cylinder(oracle) => oracle.project(query),
        }
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        match self {
            Self::Cuboid(oracle) => oracle.signed_distance(query),
            Self::Sphere(oracle) => oracle.signed_distance(query),
            Self::Cylinder(oracle) => oracle.signed_distance(query),
        }
    }
}
