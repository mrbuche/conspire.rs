#[cfg(test)]
mod test;

mod patch;

use super::{
    Brep, D, Face,
    inside::point_in_polygon,
    planar::PlanarFace,
    surface::{self, Surface},
};
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::{Cone, Cylinder, Sphere, Torus},
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};
use patch::{Analytic, Curved, FacePatch};
use std::array::from_fn;

/// [`SolidOracle`] backed by the analytic B-rep: closest-point projection onto
/// each face's exact surface. Planar faces are trimmed to their loops; curved
/// faces (cylinder, cone, sphere, torus) use an analytic primitive sized to the
/// face's extent, untrimmed in the surface parameters.
pub struct BrepOracle {
    patches: Vec<FacePatch>,
}

impl Brep {
    /// An analytic [`SolidOracle`] projecting onto this solid's surface, for
    /// fitting a background mesh. Errs on a B-spline face.
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
            Surface::Sphere(surface) => self.sphere_patch(surface, face.forward),
            Surface::Torus(surface) => self.torus_patch(surface, face.forward),
            Surface::BSpline(_) => Err("B-spline faces are not yet meshable"),
        }
    }

    /// World-space corners of every vertex on `face`'s loops.
    fn face_vertices(&self, face: &Face) -> Vec<[Scalar; D]> {
        let mut points = Vec::new();
        for bound in &face.bounds {
            if let Ok(ring) = bound.vertices(&self.edges) {
                for vertex in ring {
                    points.push(from_fn(|k| self.vertices[vertex][k].value()));
                }
            }
        }
        if points.is_empty() {
            points.extend(
                self.vertices
                    .iter()
                    .map(|vertex| from_fn(|k| vertex[k].value())),
            );
        }
        points
    }

    fn cylinder_patch(
        &self,
        surface: &surface::Cylinder,
        face: &Face,
    ) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let (low, high) = axial_span(&self.face_vertices(face), origin, axis);
        let base: Coordinate<D> = from_fn(|k| origin[k] + low * axis[k]).into();
        let cylinder = Cylinder::new(base, surface.axis.clone(), surface.radius, high - low)?;
        Ok(curved(
            Analytic::Cylinder(cylinder.oracle()?),
            &cylinder,
            face.forward,
        ))
    }

    fn cone_patch(&self, surface: &surface::Cone, face: &Face) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let (low, high) = axial_span(&self.face_vertices(face), origin, axis);
        let slope = surface.semi_angle.tan();
        let base_radius = (surface.radius + low * slope).max(1.0e-6);
        let tip_radius = (surface.radius + high * slope).max(1.0e-6);
        let base: Coordinate<D> = from_fn(|k| origin[k] + low * axis[k]).into();
        let cone = Cone::new(base, surface.axis.clone(), base_radius, tip_radius, high - low)?;
        Ok(curved(Analytic::Cone(cone.oracle()?), &cone, face.forward))
    }

    fn sphere_patch(
        &self,
        surface: &surface::Sphere,
        forward: bool,
    ) -> Result<FacePatch, &'static str> {
        let sphere = Sphere::new(surface.origin.clone(), surface.radius)?;
        Ok(curved(Analytic::Sphere(sphere.oracle()?), &sphere, forward))
    }

    fn torus_patch(
        &self,
        surface: &surface::Torus,
        forward: bool,
    ) -> Result<FacePatch, &'static str> {
        let torus = Torus::new(
            surface.origin.clone(),
            surface.axis.clone(),
            surface.major_radius,
            surface.minor_radius,
        )?;
        Ok(curved(Analytic::Torus(torus.oracle()?), &torus, forward))
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
            .filter_map(|patch| patch.closest(query))
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

/// A [`Curved`] patch from an analytic primitive and its own bounding box.
fn curved(analytic: Analytic, primitive: &impl Solid, forward: bool) -> FacePatch {
    let (low, high) = primitive
        .bounding_box()
        .unwrap_or_else(|_| ([0.0; D].into(), [0.0; D].into()));
    FacePatch::Curved(Curved {
        analytic,
        sign: if forward { 1.0 } else { -1.0 },
        low: from_fn(|k| low[k].value()),
        high: from_fn(|k| high[k].value()),
    })
}

/// The `[low, high]` span of `points` projected onto `axis` from `origin`,
/// widened to a non-degenerate interval.
fn axial_span(points: &[[Scalar; D]], origin: [Scalar; D], axis: [Scalar; D]) -> (Scalar, Scalar) {
    let mut low = Scalar::INFINITY;
    let mut high = Scalar::NEG_INFINITY;
    for point in points {
        let along = (0..D).map(|k| (point[k] - origin[k]) * axis[k]).sum::<Scalar>();
        low = low.min(along);
        high = high.max(along);
    }
    if !(low.is_finite() && high < Scalar::INFINITY) {
        (low, high) = (-0.5, 0.5);
    }
    if high - low < 1.0e-9 {
        (low - 0.5, high + 0.5)
    } else {
        (low, high)
    }
}

/// The closest point of a trimmed planar face to `query`.
pub(super) fn closest_on_face(face: &PlanarFace, query: &Coordinate<D>) -> Coordinate<D> {
    let uv = face.project(query);
    let uv = if point_in_polygon(uv, &face.rings) {
        uv
    } else {
        closest_on_rings(&face.rings, uv)
    };
    face.unproject(uv)
}

fn closest_on_rings(rings: &[Vec<[Scalar; 2]>], [px, py]: [Scalar; 2]) -> [Scalar; 2] {
    let mut best = [px, py];
    let mut best_distance = Scalar::INFINITY;
    for ring in rings {
        let count = ring.len();
        for i in 0..count {
            let [ax, ay] = ring[i];
            let [bx, by] = ring[(i + 1) % count];
            let (ex, ey) = (bx - ax, by - ay);
            let span = ex * ex + ey * ey;
            let t = if span > 0.0 {
                (((px - ax) * ex + (py - ay) * ey) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (cx, cy) = (ax + t * ex, ay + t * ey);
            let distance = (px - cx).powi(2) + (py - cy).powi(2);
            if distance < best_distance {
                best_distance = distance;
                best = [cx, cy];
            }
        }
    }
    best
}
