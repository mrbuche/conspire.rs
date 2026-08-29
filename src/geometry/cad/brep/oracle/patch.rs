//! One B-rep face as an analytic closest-point patch. Cylindrical and conical
//! faces are trimmed exactly from their bounding edges: an axial ruling or a
//! circular arc ⊥ the axis collapses to a straight segment in the surface's
//! own (angle, axial distance) chart, and a tilted elliptical edge (cylinder
//! only) traces an exact sinusoid there — no tessellation, no curve sampling
//! (nearest-point-on-sinusoid has no closed form, so that one case bisects to
//! an exact root instead, the same pattern as `csg::Ellipsoid`'s oracle).
//! Spherical and toroidal faces are taken whole.

use super::super::{D, planar::PlanarFace};
use crate::{
    geometry::{Coordinate, Direction},
    math::{Scalar, Tensor},
};
use std::array::from_fn;

pub(super) enum Curved {
    /// The lateral wall of a cylinder, trimmed either to `[low, high]` along
    /// `axis` (`rings: None`, a full sweep) or to an exact `(angle, axial
    /// distance)` polygon (`rings: Some`, a genuine partial sweep).
    Cylinder {
        origin: [Scalar; D],
        axis: [Scalar; D],
        radius: Scalar,
        low: Scalar,
        high: Scalar,
        rings: Option<Vec<super::Ring>>,
        sign: Scalar,
    },
    /// The lateral wall of a cone; `radius` is the surface's own radius at
    /// `origin` (`v = 0`), widening by `slope` per unit `v`.
    Cone {
        origin: [Scalar; D],
        axis: [Scalar; D],
        radius: Scalar,
        slope: Scalar,
        low: Scalar,
        high: Scalar,
        rings: Option<Vec<super::Ring>>,
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
                origin,
                axis,
                radius,
                low,
                high,
                rings,
                sign,
            } => {
                let (origin, axis, radius) = (*origin, *axis, *radius);
                let uv = clamp_uv(super::to_uv(origin, axis, q), *low, *high, rings, |_| radius);
                let direction = super::uv_direction(axis, uv[0]);
                let point = from_fn(|k| origin[k] + uv[1] * axis[k] + radius * direction[k]);
                (point, scaled(direction, *sign))
            }
            Self::Cone {
                origin,
                axis,
                radius,
                slope,
                low,
                high,
                rings,
                sign,
            } => {
                let (origin, axis, radius, slope) = (*origin, *axis, *radius, *slope);
                let radius_at = move |v: Scalar| (radius + v * slope).max(0.0);
                let uv = clamp_uv(super::to_uv(origin, axis, q), *low, *high, rings, radius_at);
                let direction = super::uv_direction(axis, uv[0]);
                let r = radius_at(uv[1]);
                let point = from_fn(|k| origin[k] + uv[1] * axis[k] + r * direction[k]);
                let normal_2d = unit2([1.0, -slope]).unwrap_or([1.0, 0.0]);
                let normal = from_fn(|k| normal_2d[0] * direction[k] + normal_2d[1] * axis[k]);
                (point, scaled(normal, *sign))
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

/// The `(u, v)` to project at: `uv` itself if it (or an adjacent turn) lies
/// inside `rings`, else the nearest boundary point of `rings`; with no `rings`
/// (a full sweep) just the axial clamp, angle unrestricted.
fn clamp_uv(
    uv: [Scalar; 2],
    low: Scalar,
    high: Scalar,
    rings: &Option<Vec<super::Ring>>,
    radius_at: impl Fn(Scalar) -> Scalar,
) -> [Scalar; 2] {
    match rings {
        None => [uv[0], uv[1].clamp(low, high)],
        Some(rings) => {
            if super::periodic_contains(uv, rings) {
                uv
            } else {
                super::periodic_nearest(uv, rings, radius_at)
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
