#[cfg(test)]
mod test;

use super::{Brep, D, planar::PlanarFace};
use crate::{
    geometry::{Coordinate, Direction},
    math::Scalar,
};

const EPSILON: Scalar = 1.0e-10;
const GRAZING: Scalar = 1.0e-4;

// dup: same jittered set as geometry::mesh::tessellation::{trim, cut}.
const DIRECTIONS: [Direction<D>; 3] = [
    Direction::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Direction::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Direction::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

/// Even-odd test of a point against a set of rings in a common 2D frame. Holes
/// fall out of the parity, whatever the winding.
pub(super) fn point_in_polygon([px, py]: [Scalar; 2], rings: &[Vec<[Scalar; 2]>]) -> bool {
    let mut inside = false;
    for ring in rings {
        let count = ring.len();
        for i in 0..count {
            let [ax, ay] = ring[i];
            let [bx, by] = ring[(i + 1) % count];
            if (ay > py) != (by > py) {
                let crossing = ax + (py - ay) / (by - ay) * (bx - ax);
                if px < crossing {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Whether `point` lies inside the solid, by casting jittered rays at the
/// pre-lifted faces and reading the sign of the nearest hit's face normal. A
/// grazing ray is inconclusive and the next one is tried.
pub(super) fn encloses(
    point: &Coordinate<D>,
    faces: &[PlanarFace],
    directions: &[Direction<D>; 3],
) -> bool {
    directions
        .iter()
        .find_map(|direction| {
            let mut nearest: Option<(Scalar, Scalar)> = None;
            for face in faces {
                let cosine = (direction * &face.normal).value();
                if cosine.abs() <= EPSILON {
                    continue;
                }
                let parameter = -face.plane_distance(point) / cosine;
                if parameter <= EPSILON {
                    continue;
                }
                let [s, t] = face.project(point);
                let hit = [
                    s + parameter * (direction * &face.u).value(),
                    t + parameter * (direction * &face.v).value(),
                ];
                if point_in_polygon(hit, &face.rings)
                    && nearest.is_none_or(|(best, _)| parameter < best)
                {
                    nearest = Some((parameter, cosine));
                }
            }
            match nearest {
                None => Some(false),
                Some((_, cosine)) => (cosine.abs() > GRAZING).then_some(cosine > 0.0),
            }
        })
        .unwrap_or(false)
}

impl Brep {
    /// Whether `point` lies inside this solid. Planar faces only.
    pub fn encloses(&self, point: &Coordinate<D>) -> Result<bool, &'static str> {
        let faces = self
            .faces
            .iter()
            .map(|face| self.planar_face(face))
            .collect::<Result<Vec<_>, _>>()?;
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        Ok(encloses(point, &faces, &directions))
    }
}
