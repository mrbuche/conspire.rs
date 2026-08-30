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
#[cfg(test)]
use std::f64::consts::TAU;

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

    /// Outward-wound triangles of a curved patch's trimmed surface for the
    /// winding-number integral; empty for a planar patch (its loops carry the
    /// contribution exactly).
    #[cfg(test)]
    pub(super) fn winding_triangles(&self) -> Vec<[[Scalar; D]; 3]> {
        match self {
            Self::Planar(_) => Vec::new(),
            Self::Curved { curved, .. } => curved.winding_triangles(),
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

    /// A coarse outward-wound triangle soup of the trimmed surface, for the
    /// winding-number integral only (its accuracy is topological, not
    /// geometric, so the sampling can be crude). Winding follows `sign`, so a
    /// hole's triangles face into the cavity.
    #[cfg(test)]
    pub(super) fn winding_triangles(&self) -> Vec<[[Scalar; D]; 3]> {
        let mut triangles = Vec::new();
        let mut quad = |p: [[Scalar; D]; 4], flip: bool| {
            let [a, b, c, d] = p;
            if flip {
                triangles.push([a, c, b]);
                triangles.push([a, d, c]);
            } else {
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        };
        let lerp = |lo: Scalar, hi: Scalar, i: usize, n: usize| lo + (hi - lo) * i as Scalar / n as Scalar;
        const NA: usize = 48;
        match self {
            Self::Cylinder { origin, axis, radius, low, high, rings, sign } => {
                let at = |u: Scalar, v: Scalar| {
                    let d = super::uv_direction(*axis, u);
                    from_fn(|k| origin[k] + v * axis[k] + radius * d[k])
                };
                ruled_surface(&mut triangles, rings, (*low, *high), *sign < 0.0, at);
            }
            Self::Cone { origin, axis, radius, slope, low, high, rings, sign } => {
                let at = |u: Scalar, v: Scalar| {
                    let r = (radius + v * slope).max(0.0);
                    let d = super::uv_direction(*axis, u);
                    from_fn(|k| origin[k] + v * axis[k] + r * d[k])
                };
                ruled_surface(&mut triangles, rings, (*low, *high), *sign < 0.0, at);
            }
            Self::Sphere { centre, radius, sign } => {
                let nt = 24;
                let (e1, e2) = super::basis([0.0, 0.0, 1.0]);
                let at = |phi: Scalar, theta: Scalar| {
                    let (ct, st) = (theta.cos(), theta.sin());
                    from_fn(|k| {
                        centre[k] + radius * (ct * (phi.cos() * e1[k] + phi.sin() * e2[k]) + st * [0.0, 0.0, 1.0][k])
                    })
                };
                for i in 0..NA {
                    let (p0, p1) = (lerp(0.0, TAU, i, NA), lerp(0.0, TAU, i + 1, NA));
                    for j in 0..nt {
                        let half = std::f64::consts::FRAC_PI_2;
                        let (t0, t1) = (lerp(-half, half, j, nt), lerp(-half, half, j + 1, nt));
                        quad([at(p0, t0), at(p1, t0), at(p1, t1), at(p0, t1)], *sign < 0.0);
                    }
                }
            }
            Self::Torus { centre, axis, major, minor, sign } => {
                let np = 24;
                let (e1, e2) = super::basis(*axis);
                let at = |phi: Scalar, psi: Scalar| {
                    let r = major + minor * psi.cos();
                    from_fn(|k| {
                        centre[k] + r * (phi.cos() * e1[k] + phi.sin() * e2[k]) + minor * psi.sin() * axis[k]
                    })
                };
                for i in 0..NA {
                    let (p0, p1) = (lerp(0.0, TAU, i, NA), lerp(0.0, TAU, i + 1, NA));
                    for j in 0..np {
                        let (s0, s1) = (lerp(0.0, TAU, j, np), lerp(0.0, TAU, j + 1, np));
                        quad([at(p0, s0), at(p1, s0), at(p1, s1), at(p0, s1)], *sign < 0.0);
                    }
                }
            }
        }
        triangles
    }
}

/// Outward-wound triangles of a ruled surface (cylinder or cone) whose points
/// are `at(angle, axial)`: a full `[0, 2π] × [low, high]` grid when `rings` is
/// `None`, else a fan-triangulation of each `(angle, axial)` trim polygon,
/// winding-agnostic (a clockwise ring in the chart is corrected). `flip`
/// reverses everything for a hole.
#[cfg(test)]
fn ruled_surface(
    triangles: &mut Vec<[[Scalar; D]; 3]>,
    rings: &Option<Vec<super::Ring>>,
    (low, high): (Scalar, Scalar),
    flip: bool,
    at: impl Fn(Scalar, Scalar) -> [Scalar; D],
) {
    let lerp = |a: Scalar, b: Scalar, t: Scalar| a + (b - a) * t;
    let Some(rings) = rings else {
        const NA: usize = 48;
        const NV: usize = 8;
        for i in 0..NA {
            let (u0, u1) = (
                TAU * i as Scalar / NA as Scalar,
                TAU * (i + 1) as Scalar / NA as Scalar,
            );
            for j in 0..NV {
                let (v0, v1) = (
                    lerp(low, high, j as Scalar / NV as Scalar),
                    lerp(low, high, (j + 1) as Scalar / NV as Scalar),
                );
                push_quad(
                    triangles,
                    [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)],
                    flip,
                );
            }
        }
        return;
    };
    let step = TAU / 36.0;
    for ring in rings {
        let count = ring.len();
        if count < 3 {
            continue;
        }
        // Densify sinusoid and wide edges so the lift stays accurate.
        let mut polygon: Vec<[Scalar; 2]> = Vec::new();
        for i in 0..count {
            let (a, kind) = ring[i];
            let (b, _) = ring[(i + 1) % count];
            polygon.push(a);
            let cuts = match kind {
                Some(_) => 12,
                None => (((b[0] - a[0]).abs() / step).ceil() as usize).max(1),
            };
            for s in 1..cuts {
                let t = s as Scalar / cuts as Scalar;
                let u = lerp(a[0], b[0], t);
                let v = match kind {
                    Some(sinusoid) => sinusoid.v(u),
                    None => lerp(a[1], b[1], t),
                };
                polygon.push([u, v]);
            }
        }
        let n = polygon.len();
        if n < 3 {
            continue;
        }
        let area: Scalar = (0..n)
            .map(|i| {
                let p = polygon[i];
                let q = polygon[(i + 1) % n];
                p[0] * q[1] - q[0] * p[1]
            })
            .sum();
        let reverse = flip ^ (area < 0.0);
        let centroid = [
            polygon.iter().map(|p| p[0]).sum::<Scalar>() / n as Scalar,
            polygon.iter().map(|p| p[1]).sum::<Scalar>() / n as Scalar,
        ];
        let apex = at(centroid[0], centroid[1]);
        for i in 0..n {
            let p = at(polygon[i][0], polygon[i][1]);
            let q = at(polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]);
            triangles.push(if reverse { [apex, q, p] } else { [apex, p, q] });
        }
    }
}

#[cfg(test)]
fn push_quad(triangles: &mut Vec<[[Scalar; D]; 3]>, p: [[Scalar; D]; 4], flip: bool) {
    let [a, b, c, d] = p;
    if flip {
        triangles.push([a, c, b]);
        triangles.push([a, d, c]);
    } else {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
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
