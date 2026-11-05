//! A solid that is an analytic primitive, a closed surface, or the union of solids that are, so
//! that a single [`Partition`](crate::geometry::solid::Partition) can mix them without trait
//! objects.

#[cfg(test)]
mod test;

use super::{Primitive, ops::UnionAll};
use crate::{
    geometry::{
        Coordinate,
        mesh::Watertight,
        solid::{Projection, SolidOracle},
    },
    math::Quantity,
    units::Length,
};

/// A solid known through its signed distance.
pub enum Volume {
    /// An analytic primitive.
    Primitive(Primitive),
    /// The inside of a closed triangulated surface.
    Surface(Box<Watertight>),
    /// Solids that make one, which may themselves be any of these.
    Union(UnionAll<Volume>),
}

impl Volume {
    /// The union of the solids, as one solid.
    pub fn union(volumes: Vec<Volume>) -> Self {
        Self::Union(UnionAll::new(volumes))
    }
}

impl<P: Into<Primitive>> From<P> for Volume {
    fn from(primitive: P) -> Self {
        Self::Primitive(primitive.into())
    }
}

impl From<Watertight> for Volume {
    fn from(surface: Watertight) -> Self {
        Self::Surface(Box::new(surface))
    }
}

impl SolidOracle for Volume {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        match self {
            Self::Primitive(solid) => solid.signed_distance(point),
            Self::Surface(solid) => solid.signed_distance(point),
            Self::Union(solid) => solid.signed_distance(point),
        }
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        match self {
            Self::Primitive(solid) => solid.project(point),
            Self::Surface(solid) => solid.project(point),
            Self::Union(solid) => solid.project(point),
        }
    }
}
