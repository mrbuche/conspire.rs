//! One B-rep face as a closest-point patch: a trimmed plane, or an analytic
//! curved surface sized to the face's extent.

use super::{D, super::planar::PlanarFace};
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::{ConeOracle, CylinderOracle, SphereOracle, TorusOracle},
        solid::SolidOracle,
    },
    math::{Scalar, Tensor},
};

pub(super) enum Analytic {
    Cylinder(CylinderOracle),
    Cone(ConeOracle),
    Sphere(SphereOracle),
    Torus(TorusOracle),
}

impl Analytic {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        match self {
            Self::Cylinder(oracle) => oracle.project(query),
            Self::Cone(oracle) => oracle.project(query),
            Self::Sphere(oracle) => oracle.project(query),
            Self::Torus(oracle) => oracle.project(query),
        }
    }
}

/// A curved face: an analytic surface plus the sign that turns its outward
/// normal into the outward-from-solid normal, and the face's world bounds.
pub(super) struct Curved {
    pub(super) analytic: Analytic,
    pub(super) sign: Scalar,
    pub(super) low: [Scalar; D],
    pub(super) high: [Scalar; D],
}

pub(super) enum FacePatch {
    Planar(PlanarFace),
    Curved(Curved),
}

impl FacePatch {
    /// The closest surface point to `query`, its outward-from-solid unit normal,
    /// and the distance.
    pub(super) fn closest(
        &self,
        query: &Coordinate<D>,
    ) -> Option<(Coordinate<D>, Direction<D>, Scalar)> {
        match self {
            Self::Planar(face) => {
                let point = super::closest_on_face(face, query);
                let distance = (&point - query).norm().value();
                Some((point, face.normal.clone(), distance))
            }
            Self::Curved(curved) => {
                let (point, normal) = curved.analytic.project(query)?;
                let normal = if curved.sign < 0.0 {
                    Direction::const_from(std::array::from_fn(|k| -normal[k].value()))
                } else {
                    normal
                };
                let distance = (&point - query).norm().value();
                Some((point, normal, distance))
            }
        }
    }

    /// `(low, high)` world corners bounding this face.
    pub(super) fn bounds(&self) -> ([Scalar; D], [Scalar; D]) {
        match self {
            Self::Planar(face) => (
                std::array::from_fn(|k| face.aabb.minimum()[k].value()),
                std::array::from_fn(|k| face.aabb.maximum()[k].value()),
            ),
            Self::Curved(curved) => (curved.low, curved.high),
        }
    }
}
