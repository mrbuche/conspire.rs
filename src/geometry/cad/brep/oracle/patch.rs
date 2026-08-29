//! One B-rep face as an analytic closest-point patch. Cylindrical and conical
//! faces are accepted only when they sweep the full circle, so the projection
//! onto the lateral surface needs no angular trim; the axial extent is trimmed
//! to the face's vertices. Spherical and toroidal faces are taken whole.

use super::super::{D, planar::PlanarFace};
use crate::{
    geometry::{Coordinate, Direction},
    math::{Scalar, Tensor},
};
use std::array::from_fn;

pub(super) enum Curved {
    /// The lateral wall of a finite cylinder, axial coordinate in `[0, height]`
    /// from `base` along the unit `axis`.
    Cylinder {
        base: [Scalar; D],
        axis: [Scalar; D],
        radius: Scalar,
        height: Scalar,
        sign: Scalar,
    },
    /// The lateral wall of a truncated cone.
    Cone {
        base: [Scalar; D],
        axis: [Scalar; D],
        base_radius: Scalar,
        tip_radius: Scalar,
        height: Scalar,
        sign: Scalar,
    },
    Sphere {
        centre: [Scalar; D],
        radius: Scalar,
        sign: Scalar,
    },
    Torus {
        centre: [Scalar; D],
        axis: [Scalar; D],
        major: Scalar,
        minor: Scalar,
        sign: Scalar,
    },
}

pub(super) enum FacePatch {
    Planar(PlanarFace),
    Curved { curved: Curved, low: [Scalar; D], high: [Scalar; D] },
}

impl FacePatch {
    /// Closest surface point to `query`, its outward-from-solid unit normal, and
    /// the distance.
    pub(super) fn closest(
        &self,
        query: &Coordinate<D>,
    ) -> (Coordinate<D>, Direction<D>, Scalar) {
        let q: [Scalar; D] = from_fn(|k| query[k].value());
        let (point, normal) = match self {
            Self::Planar(face) => {
                let uv = face.project(query);
                let uv = if face.contains(uv) {
                    uv
                } else {
                    face.nearest_boundary(uv)
                };
                let point = face.unproject(uv);
                return (
                    point.clone(),
                    face.normal.clone(),
                    (&point - query).norm().value(),
                );
            }
            Self::Curved { curved, .. } => curved.closest(q),
        };
        let distance = (0..D).map(|k| (point[k] - q[k]).powi(2)).sum::<Scalar>().sqrt();
        (point.into(), Direction::const_from(normal), distance)
    }

    pub(super) fn bounds(&self) -> ([Scalar; D], [Scalar; D]) {
        match self {
            Self::Planar(face) => (
                from_fn(|k| face.aabb.minimum()[k].value()),
                from_fn(|k| face.aabb.maximum()[k].value()),
            ),
            Self::Curved { low, high, .. } => (*low, *high),
        }
    }
}

impl Curved {
    fn closest(&self, q: [Scalar; D]) -> ([Scalar; D], [Scalar; D]) {
        match self {
            Self::Cylinder {
                base,
                axis,
                radius,
                height,
                sign,
            } => {
                let (h, radial, _) = local(*base, *axis, q);
                let radial_unit = unit(radial).unwrap_or_else(|| perpendicular(*axis));
                let h = h.clamp(0.0, *height);
                let point = from_fn(|k| base[k] + h * axis[k] + radius * radial_unit[k]);
                (point, scaled(radial_unit, *sign))
            }
            Self::Cone {
                base,
                axis,
                base_radius,
                tip_radius,
                height,
                sign,
            } => {
                let (h, radial, rho) = local(*base, *axis, q);
                let radial_unit = unit(radial).unwrap_or_else(|| perpendicular(*axis));
                // Project `(rho, h)` onto the slant `(base_radius, 0) -> (tip_radius, height)`.
                let d = [tip_radius - base_radius, *height];
                let span = (d[0] * d[0] + d[1] * d[1]).max(1.0e-30);
                let t = (((rho - base_radius) * d[0] + h * d[1]) / span).clamp(0.0, 1.0);
                let (radius_at, h_at) = (base_radius + t * d[0], t * d[1]);
                let point = from_fn(|k| base[k] + h_at * axis[k] + radius_at * radial_unit[k]);
                let normal_2d =
                    unit2([d[1], -d[0]]).unwrap_or([1.0, 0.0]);
                let normal = from_fn(|k| normal_2d[0] * radial_unit[k] + normal_2d[1] * axis[k]);
                (point, scaled(unit(normal).unwrap_or(radial_unit), *sign))
            }
            Self::Sphere {
                centre,
                radius,
                sign,
            } => {
                let delta: [Scalar; D] = from_fn(|k| q[k] - centre[k]);
                let normal = unit(delta).unwrap_or([1.0, 0.0, 0.0]);
                (from_fn(|k| centre[k] + radius * normal[k]), scaled(normal, *sign))
            }
            Self::Torus {
                centre,
                axis,
                major,
                minor,
                sign,
            } => {
                let (_, radial, _) = local(*centre, *axis, q);
                let radial_unit = unit(radial).unwrap_or_else(|| perpendicular(*axis));
                let ring: [Scalar; D] = from_fn(|k| centre[k] + major * radial_unit[k]);
                let off: [Scalar; D] = from_fn(|k| q[k] - ring[k]);
                let normal = unit(off).unwrap_or(radial_unit);
                (from_fn(|k| ring[k] + minor * normal[k]), scaled(normal, *sign))
            }
        }
    }
}

fn local(base: [Scalar; D], axis: [Scalar; D], q: [Scalar; D]) -> (Scalar, [Scalar; D], Scalar) {
    let rel: [Scalar; D] = from_fn(|k| q[k] - base[k]);
    let h = (0..D).map(|k| rel[k] * axis[k]).sum::<Scalar>();
    let radial: [Scalar; D] = from_fn(|k| rel[k] - h * axis[k]);
    let rho = radial.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (h, radial, rho)
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 1.0e-30).then(|| v.map(|x| x / norm))
}

fn unit2(v: [Scalar; 2]) -> Option<[Scalar; 2]> {
    let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
    (norm > 1.0e-30).then(|| [v[0] / norm, v[1] / norm])
}

fn scaled(v: [Scalar; D], sign: Scalar) -> [Scalar; D] {
    v.map(|x| sign * x)
}

fn perpendicular(axis: [Scalar; D]) -> [Scalar; D] {
    let seed = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot = (0..D).map(|k| seed[k] * axis[k]).sum::<Scalar>();
    unit(from_fn(|k| seed[k] - dot * axis[k])).unwrap_or([1.0, 0.0, 0.0])
}
