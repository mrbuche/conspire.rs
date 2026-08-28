#[cfg(test)]
mod test;

use super::{Brep, D, Face, surface::Surface};
use crate::{
    geometry::{Coordinate, CoordinatesRef, Direction, bbox::BoundingBox},
    math::{CrossProduct, Scalar, Tensor},
};
use std::array::from_fn;

const EPSILON: Scalar = 1e-10;

/// A planar B-rep face lifted into its own 2D frame: an orthonormal in-plane
/// basis `(u, v)` with `v = normal x u`, the outward unit normal, and every
/// bounding loop projected into that frame with the outer loop first, then
/// holes.
pub struct PlanarFace {
    pub origin: Coordinate<D>,
    pub normal: Direction<D>,
    pub u: Direction<D>,
    pub v: Direction<D>,
    pub rings: Vec<Vec<[Scalar; 2]>>,
    /// The outer loop's vertices in world space, for box-overlap tests.
    pub outline: Vec<[Scalar; D]>,
    pub aabb: BoundingBox<D>,
}

impl PlanarFace {
    /// `point` dropped into the `(u, v)` frame, measured from `origin`.
    pub fn project(&self, point: &Coordinate<D>) -> [Scalar; 2] {
        let delta = point - &self.origin;
        [(&delta * &self.u).value(), (&delta * &self.v).value()]
    }
    /// The `(u, v)` pair lifted back onto the plane.
    pub fn unproject(&self, [s, t]: [Scalar; 2]) -> Coordinate<D> {
        let coordinates: [Scalar; D] =
            from_fn(|k| self.origin[k].value() + s * self.u[k].value() + t * self.v[k].value());
        Coordinate::from(coordinates)
    }
    /// Signed distance from `point` to the plane, positive on the outward side.
    pub fn plane_distance(&self, point: &Coordinate<D>) -> Scalar {
        ((point - &self.origin) * &self.normal).value()
    }
}

impl Brep {
    pub fn planar_face(&self, face: &Face) -> Result<PlanarFace, &'static str> {
        let Surface::Plane(plane) = &face.surface else {
            return Err("planar_face called on a non-planar face");
        };
        let sign = if face.forward { 1.0 } else { -1.0 };
        let normal = (&plane.normal * sign).normalized();
        let reference = &plane.reference_direction;
        let projection = reference - &(&normal * (reference * &normal));
        if projection.norm().value() <= EPSILON {
            return Err("degenerate plane axes");
        }
        let u = projection.normalized();
        let v = normal.cross(&u);

        let loops = face
            .bounds
            .iter()
            .map(|bound| bound.vertices(&self.edges))
            .collect::<Result<Vec<_>, _>>()?;
        let &first = loops
            .first()
            .and_then(|ring| ring.first())
            .ok_or("face has no outer loop")?;
        let origin = self.vertices[first].clone();

        let mut refs: Vec<&Coordinate<D>> = Vec::new();
        let mut outline = Vec::new();
        let mut rings = Vec::with_capacity(loops.len());
        for (index, ring) in loops.iter().enumerate() {
            if ring.len() < 3 {
                return Err("face loop has fewer than three vertices");
            }
            rings.push(
                ring.iter()
                    .map(|&vertex| {
                        let point = &self.vertices[vertex];
                        refs.push(point);
                        if index == 0 {
                            outline.push(from_fn(|k| point[k].value()));
                        }
                        let delta = point - &origin;
                        [(&delta * &u).value(), (&delta * &v).value()]
                    })
                    .collect(),
            );
        }
        let aabb = BoundingBox::from(refs.into_iter().collect::<CoordinatesRef<'_, D>>());

        Ok(PlanarFace {
            origin,
            normal,
            u,
            v,
            rings,
            outline,
            aabb,
        })
    }
}
