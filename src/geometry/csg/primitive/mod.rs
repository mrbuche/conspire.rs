//! A closed enum over the analytic primitives, so a single [`Union`](super::ops::Union) or
//! [`Partition`](crate::geometry::solid::Partition) can mix shapes without trait objects.

#[cfg(test)]
mod test;

use super::{Cone, Cuboid, Cylinder, Ellipsoid, Sphere, Torus};
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
    Cone(Cone),
    Cuboid(Cuboid),
    Cylinder(Cylinder),
    Ellipsoid(Ellipsoid),
    Sphere(Sphere),
    Torus(Torus),
}

impl From<Cone> for Primitive {
    fn from(primitive: Cone) -> Self {
        Self::Cone(primitive)
    }
}

impl From<Cuboid> for Primitive {
    fn from(primitive: Cuboid) -> Self {
        Self::Cuboid(primitive)
    }
}

impl From<Cylinder> for Primitive {
    fn from(primitive: Cylinder) -> Self {
        Self::Cylinder(primitive)
    }
}

impl From<Ellipsoid> for Primitive {
    fn from(primitive: Ellipsoid) -> Self {
        Self::Ellipsoid(primitive)
    }
}

impl From<Sphere> for Primitive {
    fn from(primitive: Sphere) -> Self {
        Self::Sphere(primitive)
    }
}

impl From<Torus> for Primitive {
    fn from(primitive: Torus) -> Self {
        Self::Torus(primitive)
    }
}

impl SolidOracle for Primitive {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        match self {
            Self::Cone(primitive) => primitive.signed_distance(point),
            Self::Cuboid(primitive) => primitive.signed_distance(point),
            Self::Cylinder(primitive) => primitive.signed_distance(point),
            Self::Ellipsoid(primitive) => primitive.signed_distance(point),
            Self::Sphere(primitive) => primitive.signed_distance(point),
            Self::Torus(primitive) => primitive.signed_distance(point),
        }
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        match self {
            Self::Cone(primitive) => primitive.project(point),
            Self::Cuboid(primitive) => primitive.project(point),
            Self::Cylinder(primitive) => primitive.project(point),
            Self::Ellipsoid(primitive) => primitive.project(point),
            Self::Sphere(primitive) => primitive.project(point),
            Self::Torus(primitive) => primitive.project(point),
        }
    }
}
