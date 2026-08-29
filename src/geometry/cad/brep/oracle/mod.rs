#[cfg(test)]
mod test;

mod patch;

use super::{
    Brep, D, Face,
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
/// or exact disk/annulus). Cylindrical and conical faces are accepted only when
/// they sweep the full circle — a partial one (fillet, chamfer) errs — and are
/// trimmed to the face's axial vertex span. Spherical and toroidal faces are
/// taken whole. A B-spline face errs.
pub struct BrepOracle {
    patches: Vec<FacePatch>,
}

/// A cylindrical or conical face whose vertices span more than this fraction of
/// a turn away from complete is treated as a partial patch and rejected.
const ANGULAR_GAP_LIMIT: Scalar = std::f64::consts::FRAC_PI_2;

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

    fn cylinder_patch(
        &self,
        surface: &surface::Cylinder,
        face: &Face,
    ) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        if !sweeps_full_circle(&vertices, origin, axis) {
            return Err("partial cylindrical face (fillet/chamfer) is not yet meshable");
        }
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let radius = surface.radius;
        let curved = Curved::Cylinder {
            base,
            axis,
            radius,
            height: high - low,
            sign: orientation(face.forward),
        };
        let (bl, bh) = frustum_bounds(base, axis, radius, radius, high - low);
        Ok(FacePatch::Curved { curved, low: bl, high: bh })
    }

    fn cone_patch(&self, surface: &surface::Cone, face: &Face) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        if !sweeps_full_circle(&vertices, origin, axis) {
            return Err("partial conical face (fillet/chamfer) is not yet meshable");
        }
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let slope = surface.semi_angle.tan();
        let base_radius = (surface.radius + low * slope).max(0.0);
        let tip_radius = (surface.radius + high * slope).max(0.0);
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let curved = Curved::Cone {
            base,
            axis,
            base_radius,
            tip_radius,
            height: high - low,
            sign: orientation(face.forward),
        };
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

/// Whether `points` projected around `axis` cover the whole circle: fewer than
/// two distinct angles (a seam), or the largest wrap-around gap between sorted
/// angles is small enough that no chunk of the turn is missing.
fn sweeps_full_circle(points: &[[Scalar; D]], origin: [Scalar; D], axis: [Scalar; D]) -> bool {
    let (u, v) = basis(axis);
    let mut angles: Vec<Scalar> = points
        .iter()
        .filter_map(|point| {
            let rel: [Scalar; D] = from_fn(|k| point[k] - origin[k]);
            let (s, t) = (dot(rel, u), dot(rel, v));
            (s.hypot(t) > 1.0e-9).then(|| t.atan2(s))
        })
        .collect();
    angles.sort_by(Scalar::total_cmp);
    angles.dedup_by(|a, b| (*a - *b).abs() < 1.0e-6);
    // A seam sitting on the branch cut lands at both -pi and +pi: one angle.
    if angles.len() > 1 && angles[0] + std::f64::consts::TAU - angles[angles.len() - 1] < 1.0e-6 {
        angles.pop();
    }
    if angles.len() < 2 {
        return true;
    }
    let mut gap = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
    for pair in angles.windows(2) {
        gap = gap.max(pair[1] - pair[0]);
    }
    gap <= ANGULAR_GAP_LIMIT
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
