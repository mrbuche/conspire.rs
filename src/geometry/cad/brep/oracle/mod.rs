#[cfg(test)]
mod test;

use super::{
    Brep, D,
    inside::{directions, encloses, point_in_polygon},
    planar::PlanarFace,
};
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::{Scalar, Tensor},
};

/// [`SolidOracle`] backed by the analytic B-rep: exact closest-point projection
/// onto the trimmed planar faces, returning the hit face's exact plane normal.
/// No tessellation, no facet normals.
pub struct BrepOracle {
    faces: Vec<PlanarFace>,
    directions: [Direction<D>; 3],
}

impl Brep {
    /// An analytic [`Oracle`] projecting onto this solid's exact surface, for
    /// fitting a background mesh to the geometry. Planar faces only.
    pub fn oracle(&self) -> Result<BrepOracle, &'static str> {
        if self.faces.is_empty() {
            return Err("brep has no faces");
        }
        Ok(BrepOracle {
            faces: self
                .faces
                .iter()
                .map(|face| self.planar_face(face))
                .collect::<Result<Vec<_>, _>>()?,
            directions: directions(),
        })
    }
}

impl SolidOracle for BrepOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        self.faces
            .iter()
            .map(|face| {
                let point = closest_on_face(face, query);
                let distance = (&point - query).norm().value();
                (point, face.normal.clone(), distance)
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(point, normal, _)| (point, normal))
    }

    /// The magnitude is the exact distance to the nearest trimmed face; the sign
    /// is positive when `query` is enclosed by the solid.
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let magnitude = self
            .faces
            .iter()
            .map(|face| (&closest_on_face(face, query) - query).norm().value())
            .fold(Scalar::INFINITY, Scalar::min);
        if encloses(query, &self.faces, &self.directions) {
            magnitude
        } else {
            -magnitude
        }
    }
}

/// The closest point of a trimmed planar face to `query`: the foot of the
/// perpendicular when it lands inside the trimming loops, otherwise the nearest
/// point on their boundary.
fn closest_on_face(face: &PlanarFace, query: &Coordinate<D>) -> Coordinate<D> {
    let uv = face.project(query);
    let uv = if point_in_polygon(uv, &face.rings) {
        uv
    } else {
        closest_on_rings(&face.rings, uv)
    };
    face.unproject(uv)
}

/// The point on the union of ring polylines nearest to `[px, py]`, all in a
/// common 2D frame.
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
