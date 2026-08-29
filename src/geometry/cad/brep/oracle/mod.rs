#[cfg(test)]
mod test;

mod patch;

use super::{
    Brep, D, Face, Loop,
    curve::Curve,
    surface::{self, Surface},
};
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::Scalar,
};
use patch::{Curved, FacePatch};
use std::array::from_fn;

/// [`SolidOracle`] backed by the analytic B-rep: closest-point projection onto
/// each face's exact surface. Planar faces are trimmed to their loops (polygon
/// or exact disk/annulus). Cylindrical and conical faces are trimmed exactly
/// from their bounding edges — an axial ruling or a circular arc ⊥ the axis
/// collapses to a straight segment in the surface's own (angle, axial
/// distance) chart, so a genuine partial sweep (fillet, chamfer remnant) is
/// meshable; a tilted or free-form edge on such a face still errs. Spherical
/// and toroidal faces are taken whole. A B-spline face errs.
pub struct BrepOracle {
    patches: Vec<FacePatch>,
}

impl Brep {
    /// An analytic [`SolidOracle`] projecting onto this solid's surface, for
    /// fitting a background mesh.
    pub fn oracle(&self) -> Result<BrepOracle, &'static str> {
        if self.faces.is_empty() {
            return Err("brep has no faces");
        }
        Ok(BrepOracle {
            patches: self
                .faces
                .iter()
                .map(|face| self.face_patch(face))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn face_patch(&self, face: &Face) -> Result<FacePatch, &'static str> {
        match &face.surface {
            Surface::Plane(_) => Ok(FacePatch::Planar(self.planar_face(face)?)),
            Surface::Cylinder(surface) => self.cylinder_patch(surface, face),
            Surface::Cone(surface) => self.cone_patch(surface, face),
            Surface::Sphere(surface) => Ok(sphere_patch(surface, face.forward)),
            Surface::Torus(surface) => Ok(torus_patch(surface, face.forward)),
            Surface::BSpline(_) => Err("B-spline faces are not yet meshable"),
        }
    }

    /// World-space corners of every vertex on `face`'s loops, poles included.
    fn face_vertices(&self, face: &Face) -> Result<Vec<[Scalar; D]>, &'static str> {
        let mut points = Vec::new();
        for bound in &face.bounds {
            for vertex in bound.vertices(&self.edges)? {
                points.push(from_fn(|k| self.vertices[vertex][k].value()));
            }
        }
        for &pole in &face.poles {
            points.push(from_fn(|k| self.vertices[pole][k].value()));
        }
        Ok(points)
    }

    /// The `(u, v)` boundary of one bound in `origin`/`axis`'s chart —
    /// `u` the angle around `axis`, unwrapped continuously past a single turn
    /// as the loop is walked, `v` the signed axial distance — or `None` if the
    /// bound has no angular restriction (a coincident-endpoint seam circle,
    /// the whole-turn case).
    fn uv_ring(
        &self,
        bound: &Loop,
        origin: [Scalar; D],
        axis: [Scalar; D],
    ) -> Result<Option<Vec<[Scalar; 2]>>, &'static str> {
        let mut ring: Vec<[Scalar; 2]> = Vec::new();
        for half_edge in &bound.half_edges {
            let edge = self
                .edges
                .get(half_edge.edge)
                .ok_or("half-edge references a missing edge")?;
            let (start, end) = if half_edge.forward {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            if ring.is_empty() {
                let start_point: [Scalar; D] = from_fn(|k| self.vertices[start][k].value());
                ring.push(to_uv(origin, axis, start_point));
            }
            let [u_prev, _] = *ring.last().unwrap();
            let end_point: [Scalar; D] = from_fn(|k| self.vertices[end][k].value());
            match &edge.curve {
                Curve::Line(_) => {
                    let [u_end, v_end] = to_uv(origin, axis, end_point);
                    if wrap(u_end - u_prev).abs() > 1.0e-6 {
                        return Err("non-axial straight edge on a curved face");
                    }
                    ring.push([u_prev, v_end]);
                }
                Curve::Circle(circle) => {
                    let caxis: [Scalar; D] = from_fn(|k| circle.axis[k].value());
                    let alignment = dot(caxis, axis);
                    if alignment.abs() < 1.0 - 1.0e-6 {
                        return Err("tilted circular edge on a curved face");
                    }
                    if start == end {
                        return Ok(None);
                    }
                    let sign = if half_edge.forward { alignment } else { -alignment };
                    let [u_end, v_end] = to_uv(origin, axis, end_point);
                    let mut delta = wrap(u_end - u_prev);
                    if sign > 0.0 && delta < 0.0 {
                        delta += std::f64::consts::TAU;
                    } else if sign < 0.0 && delta > 0.0 {
                        delta -= std::f64::consts::TAU;
                    }
                    ring.push([u_prev + delta, v_end]);
                }
                _ => return Err("unsupported edge on a curved face trim"),
            }
        }
        Ok(Some(ring))
    }

    /// Every bound of `face` as a `(u, v)` polygon, or `None` if any bound
    /// sweeps the whole turn unrestricted.
    fn trim_rings(
        &self,
        face: &Face,
        origin: [Scalar; D],
        axis: [Scalar; D],
    ) -> Result<Option<Vec<Vec<[Scalar; 2]>>>, &'static str> {
        let mut rings = Vec::new();
        for bound in &face.bounds {
            match self.uv_ring(bound, origin, axis)? {
                Some(ring) => rings.push(ring),
                None => return Ok(None),
            }
        }
        Ok(Some(rings))
    }

    fn cylinder_patch(
        &self,
        surface: &surface::Cylinder,
        face: &Face,
    ) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let radius = surface.radius;
        let rings = self.trim_rings(face, origin, axis)?;
        let curved = Curved::Cylinder {
            origin,
            axis,
            radius,
            low,
            high,
            rings,
            sign: orientation(face.forward),
        };
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let (bl, bh) = frustum_bounds(base, axis, radius, radius, high - low);
        Ok(FacePatch::Curved { curved, low: bl, high: bh })
    }

    fn cone_patch(&self, surface: &surface::Cone, face: &Face) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let slope = surface.semi_angle.tan();
        let base_radius = (surface.radius + low * slope).max(0.0);
        let tip_radius = (surface.radius + high * slope).max(0.0);
        let rings = self.trim_rings(face, origin, axis)?;
        let curved = Curved::Cone {
            origin,
            axis,
            radius: surface.radius,
            slope,
            low,
            high,
            rings,
            sign: orientation(face.forward),
        };
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let (bl, bh) = frustum_bounds(base, axis, base_radius, tip_radius, high - low);
        Ok(FacePatch::Curved { curved, low: bl, high: bh })
    }
}

fn sphere_patch(surface: &surface::Sphere, forward: bool) -> FacePatch {
    let centre: [Scalar; D] = from_fn(|k| surface.origin[k].value());
    let radius = surface.radius;
    FacePatch::Curved {
        curved: Curved::Sphere {
            centre,
            radius,
            sign: orientation(forward),
        },
        low: from_fn(|k| centre[k] - radius),
        high: from_fn(|k| centre[k] + radius),
    }
}

fn torus_patch(surface: &surface::Torus, forward: bool) -> FacePatch {
    let centre: [Scalar; D] = from_fn(|k| surface.origin[k].value());
    let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
    let (major, minor) = (surface.major_radius, surface.minor_radius);
    let reach: [Scalar; D] =
        from_fn(|k| (major + minor) * (1.0 - axis[k] * axis[k]).max(0.0).sqrt() + minor * axis[k].abs());
    FacePatch::Curved {
        curved: Curved::Torus {
            centre,
            axis,
            major,
            minor,
            sign: orientation(forward),
        },
        low: from_fn(|k| centre[k] - reach[k]),
        high: from_fn(|k| centre[k] + reach[k]),
    }
}

impl BrepOracle {
    /// `(low, high)` world corners enclosing every face.
    pub fn bounds(&self) -> (Coordinate<D>, Coordinate<D>) {
        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for patch in &self.patches {
            let (patch_low, patch_high) = patch.bounds();
            for k in 0..D {
                low[k] = low[k].min(patch_low[k]);
                high[k] = high[k].max(patch_high[k]);
            }
        }
        (low.into(), high.into())
    }

    fn nearest(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>, Scalar)> {
        self.patches
            .iter()
            .map(|patch| patch.closest(query))
            .min_by(|a, b| a.2.total_cmp(&b.2))
    }
}

impl SolidOracle for BrepOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        self.nearest(query).map(|(point, normal, _)| (point, normal))
    }

    /// Magnitude is the distance to the nearest face; the sign is read from that
    /// face's outward normal (positive inside).
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        match self.nearest(query) {
            Some((point, normal, distance)) => {
                let outward: Scalar = (0..D)
                    .map(|k| (query[k].value() - point[k].value()) * normal[k].value())
                    .sum();
                if outward >= 0.0 { -distance } else { distance }
            }
            None => Scalar::NEG_INFINITY,
        }
    }
}

fn orientation(forward: bool) -> Scalar {
    if forward { 1.0 } else { -1.0 }
}

/// The `[low, high]` span of `points` projected onto `axis` from `origin`.
/// Errs rather than inventing a span when the face has no usable extent — a
/// degenerate span has no honest analytic patch.
fn axial_span(
    points: &[[Scalar; D]],
    origin: [Scalar; D],
    axis: [Scalar; D],
) -> Result<(Scalar, Scalar), &'static str> {
    let mut low = Scalar::INFINITY;
    let mut high = Scalar::NEG_INFINITY;
    for point in points {
        let along = (0..D).map(|k| (point[k] - origin[k]) * axis[k]).sum::<Scalar>();
        low = low.min(along);
        high = high.max(along);
    }
    if !(low.is_finite() && high.is_finite()) {
        return Err("cylindrical/conical face has no vertices to bound its axial extent");
    }
    if high - low < 1.0e-9 {
        return Err("cylindrical/conical face has a degenerate (zero-height) axial extent");
    }
    Ok((low, high))
}

/// AABB of a frustum: the union of its two end circles, each an exact
/// axis-aligned box of a circle with the given radius, centre and `axis`.
fn frustum_bounds(
    base: [Scalar; D],
    axis: [Scalar; D],
    base_radius: Scalar,
    tip_radius: Scalar,
    height: Scalar,
) -> ([Scalar; D], [Scalar; D]) {
    let tip: [Scalar; D] = from_fn(|k| base[k] + height * axis[k]);
    let mut low = [Scalar::INFINITY; D];
    let mut high = [Scalar::NEG_INFINITY; D];
    for (centre, radius) in [(base, base_radius), (tip, tip_radius)] {
        for k in 0..D {
            let extent = radius * (1.0 - axis[k] * axis[k]).max(0.0).sqrt();
            low[k] = low[k].min(centre[k] - extent);
            high[k] = high[k].max(centre[k] + extent);
        }
    }
    (low, high)
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

/// An orthonormal pair spanning the plane perpendicular to `axis`.
fn basis(axis: [Scalar; D]) -> ([Scalar; D], [Scalar; D]) {
    let seed = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let d = dot(seed, axis);
    let mut u: [Scalar; D] = from_fn(|k| seed[k] - d * axis[k]);
    let norm = dot(u, u).sqrt().max(1.0e-30);
    u = u.map(|x| x / norm);
    let v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ];
    (u, v)
}

/// `(u, v)` of `point` in `origin`/`axis`'s own chart: `u` the angle around
/// `axis` from an arbitrary but fixed in-plane reference, `v` the signed axial
/// distance from `origin`.
fn to_uv(origin: [Scalar; D], axis: [Scalar; D], point: [Scalar; D]) -> [Scalar; 2] {
    let (u_hat, v_hat) = basis(axis);
    let rel: [Scalar; D] = from_fn(|k| point[k] - origin[k]);
    let v = dot(rel, axis);
    let radial: [Scalar; D] = from_fn(|k| rel[k] - v * axis[k]);
    [dot(radial, v_hat).atan2(dot(radial, u_hat)), v]
}

/// The unit radial direction at angle `u` around `axis`.
fn uv_direction(axis: [Scalar; D], u: Scalar) -> [Scalar; D] {
    let (u_hat, v_hat) = basis(axis);
    from_fn(|k| u.cos() * u_hat[k] + u.sin() * v_hat[k])
}

/// `delta` reduced into `(-pi, pi]`.
fn wrap(delta: Scalar) -> Scalar {
    let mut d = delta % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d <= -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
}

/// Whether `uv` lies inside `rings`, trying both neighbouring turns since `u`
/// is periodic and the rings may be unwrapped past a single turn.
fn periodic_contains(uv: [Scalar; 2], rings: &[Vec<[Scalar; 2]>]) -> bool {
    [0.0, std::f64::consts::TAU, -std::f64::consts::TAU]
        .into_iter()
        .any(|shift| super::inside::point_in_polygon([uv[0] + shift, uv[1]], rings))
}

/// The point of `rings` nearest `uv`, in an arc-length metric (`weight(v)` is
/// the local radius) so the snap distance is physical, not raw angle-vs-length.
/// Periodic in `u` like [`periodic_contains`].
fn periodic_nearest(
    uv: [Scalar; 2],
    rings: &[Vec<[Scalar; 2]>],
    weight: impl Fn(Scalar) -> Scalar,
) -> [Scalar; 2] {
    let mut best = uv;
    let mut best_distance = Scalar::INFINITY;
    for shift in [0.0, std::f64::consts::TAU, -std::f64::consts::TAU] {
        let query = [uv[0] + shift, uv[1]];
        let wq = [weight(query[1]) * query[0], query[1]];
        for ring in rings {
            let count = ring.len();
            for i in 0..count {
                let a = ring[i];
                let b = ring[(i + 1) % count];
                let wa = [weight(a[1]) * a[0], a[1]];
                let wb = [weight(b[1]) * b[0], b[1]];
                let (ex, ey) = (wb[0] - wa[0], wb[1] - wa[1]);
                let span = ex * ex + ey * ey;
                let t = if span > 0.0 {
                    (((wq[0] - wa[0]) * ex + (wq[1] - wa[1]) * ey) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let candidate = [wa[0] + t * ex, wa[1] + t * ey];
                let distance = (candidate[0] - wq[0]).powi(2) + (candidate[1] - wq[1]).powi(2);
                if distance < best_distance {
                    best_distance = distance;
                    let v = candidate[1];
                    let u = candidate[0] / weight(v).max(1.0e-12) - shift;
                    best = [u, v];
                }
            }
        }
    }
    best
}
