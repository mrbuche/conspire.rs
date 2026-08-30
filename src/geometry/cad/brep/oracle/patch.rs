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

    /// Parameters `t > 0` at which the ray `origin + t·direction` crosses this
    /// patch's *trimmed* surface (`direction` need not be unit). The basis of
    /// the ray-parity inside/outside test.
    pub(super) fn ray_hits(&self, origin: [Scalar; D], direction: [Scalar; D]) -> Vec<Scalar> {
        match self {
            Self::Planar(face) => {
                let normal: [Scalar; D] = from_fn(|k| face.normal[k].value());
                let denominator = dot(direction, normal);
                if denominator.abs() < 1.0e-13 {
                    return Vec::new();
                }
                let to_plane: [Scalar; D] = from_fn(|k| face.origin[k].value() - origin[k]);
                let t = dot(to_plane, normal) / denominator;
                if t <= RAY_EPS {
                    return Vec::new();
                }
                let hit = Coordinate::from(from_fn::<Scalar, D, _>(|k| origin[k] + t * direction[k]));
                if face.contains(face.project(&hit)) {
                    vec![t]
                } else {
                    Vec::new()
                }
            }
            Self::Curved { curved, .. } => curved.ray_hits(origin, direction),
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

    /// Parameters `t > 0` where the ray `o + t·d` crosses this curved patch's
    /// trimmed surface. Quadrics (cylinder, cone, sphere) solve in closed
    /// form; the torus marches its analytic distance and bisects sign changes.
    /// Sphere and torus have no trim rings yet, so every quadric root counts.
    pub(super) fn ray_hits(&self, o: [Scalar; D], d: [Scalar; D]) -> Vec<Scalar> {
        let on_ruled = |t: &Scalar,
                        origin: &[Scalar; D],
                        axis: &[Scalar; D],
                        low: &Scalar,
                        high: &Scalar,
                        rings: &Option<Vec<super::Ring>>| {
            let hit: [Scalar; D] = from_fn(|k| o[k] + t * d[k]);
            let uv = super::to_uv(*origin, *axis, hit);
            uv[1] >= low - RAY_EPS
                && uv[1] <= high + RAY_EPS
                && rings.as_ref().is_none_or(|r| super::periodic_contains(uv, r))
        };
        match self {
            Self::Cylinder { origin, axis, radius, low, high, rings, .. } => {
                let w: [Scalar; D] = from_fn(|k| o[k] - origin[k]);
                let (dp, wp) = (reject(d, *axis), reject(w, *axis));
                quadratic(dot(dp, dp), 2.0 * dot(dp, wp), dot(wp, wp) - radius * radius)
                    .into_iter()
                    .filter(|t| *t > RAY_EPS && on_ruled(t, origin, axis, low, high, rings))
                    .collect()
            }
            Self::Cone { origin, axis, radius, slope, low, high, rings, .. } => {
                let w: [Scalar; D] = from_fn(|k| o[k] - origin[k]);
                let (dp, wp) = (reject(d, *axis), reject(w, *axis));
                let (base, rate) = (radius + slope * dot(w, *axis), slope * dot(d, *axis));
                quadratic(
                    dot(dp, dp) - rate * rate,
                    2.0 * (dot(dp, wp) - base * rate),
                    dot(wp, wp) - base * base,
                )
                .into_iter()
                .filter(|t| *t > RAY_EPS && on_ruled(t, origin, axis, low, high, rings))
                .collect()
            }
            Self::Sphere { centre, radius, .. } => {
                let w: [Scalar; D] = from_fn(|k| o[k] - centre[k]);
                quadratic(dot(d, d), 2.0 * dot(w, d), dot(w, w) - radius * radius)
                    .into_iter()
                    .filter(|t| *t > RAY_EPS)
                    .collect()
            }
            Self::Torus { centre, axis, major, minor, .. } => {
                let sdf = |p: [Scalar; D]| {
                    let (_, radial, _) = local(*centre, *axis, p);
                    let unit_radial = unit(radial).unwrap_or_else(|| perpendicular(*axis));
                    let ring: [Scalar; D] = from_fn(|k| centre[k] + major * unit_radial[k]);
                    let off: [Scalar; D] = from_fn(|k| p[k] - ring[k]);
                    dot(off, off).sqrt() - minor
                };
                march_sign_changes(o, d, *centre, major + minor, sdf)
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

const RAY_EPS: Scalar = 1.0e-12;

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

/// `v` with its component along the unit vector `axis` removed.
fn reject(v: [Scalar; D], axis: [Scalar; D]) -> [Scalar; D] {
    let along = dot(v, axis);
    from_fn(|k| v[k] - along * axis[k])
}

/// Real roots of `a·t² + b·t + c` (the lone linear root when `a ≈ 0`, none
/// when the discriminant is negative). Order is not guaranteed for `a < 0`.
fn quadratic(a: Scalar, b: Scalar, c: Scalar) -> Vec<Scalar> {
    if a.abs() < 1.0e-15 {
        return if b.abs() < 1.0e-15 {
            Vec::new()
        } else {
            vec![-c / b]
        };
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.sqrt();
    vec![(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
}

/// Ray parameters `t > 0` where `sdf(o + t·d)` changes sign, searched over the
/// segment inside the bounding sphere of radius `radius` about `centre` and
/// bisected to a root — the fallback where a closed-form ray/surface solve
/// would be a quartic or worse (the torus).
fn march_sign_changes(
    o: [Scalar; D],
    d: [Scalar; D],
    centre: [Scalar; D],
    radius: Scalar,
    sdf: impl Fn([Scalar; D]) -> Scalar,
) -> Vec<Scalar> {
    let w: [Scalar; D] = from_fn(|k| o[k] - centre[k]);
    let bracket = quadratic(dot(d, d), 2.0 * dot(w, d), dot(w, w) - radius * radius);
    let (Some(&a), Some(&b)) = (bracket.first(), bracket.last()) else {
        return Vec::new();
    };
    let (t0, t1) = (a.min(b).max(RAY_EPS), a.max(b));
    if t1 <= t0 {
        return Vec::new();
    }
    const STEPS: usize = 96;
    let at = |t: Scalar| sdf(from_fn(|k| o[k] + t * d[k]));
    let mut hits = Vec::new();
    let mut previous = at(t0);
    for i in 1..=STEPS {
        let t = t0 + (t1 - t0) * i as Scalar / STEPS as Scalar;
        let current = at(t);
        if previous != 0.0 && (previous < 0.0) != (current < 0.0) {
            let (mut lo, mut hi) = (t - (t1 - t0) / STEPS as Scalar, t);
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                if (at(mid) < 0.0) == (previous < 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hits.push(0.5 * (lo + hi));
        }
        previous = current;
    }
    hits
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
