#[cfg(test)]
mod test;

use std::mem::swap;

use crate::{
    ABS_TOL,
    geometry::{Coordinate, Direction, bbox::BoundingBox, bvh::ray::Ray},
    math::{Quantity, Scalar},
    units::{Area, Length},
};

impl<const D: usize> Ray<D> {
    pub fn origin(&self) -> &Coordinate<D> {
        &self.origin
    }
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }
    pub fn intersects(&self, bounding_box: &BoundingBox<D>) -> Option<Quantity<Length>> {
        let mut t_min = Quantity::<Length>::default();
        let mut t_max = Quantity::<Length>::new(Scalar::INFINITY);
        for axis in 0..D {
            let inverse_direction = self.inverse_direction[axis];
            let mut t_near = (bounding_box.minimum()[axis] - self.origin[axis]) * inverse_direction;
            let mut t_far = (bounding_box.maximum()[axis] - self.origin[axis]) * inverse_direction;
            if inverse_direction < 0.0 {
                swap(&mut t_near, &mut t_far)
            }
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
            if t_max < t_min {
                return None;
            }
        }
        Some(t_min)
    }
}

impl Ray<3> {
    /// Watertight ray/triangle intersection (Woop, Benthin & Wald, 2013).
    ///
    /// Unlike naive Möller–Trumbore, a ray passing exactly along a shared
    /// edge/vertex between two adjacent triangles cannot be rejected by
    /// both of them due to independent floating-point rounding: the
    /// dominant-axis permutation and shear put every triangle sharing that
    /// ray through the same, consistent 2D edge test.
    pub fn intersects_triangle(
        &self,
        a: &Coordinate<3>,
        b: &Coordinate<3>,
        c: &Coordinate<3>,
    ) -> Option<Quantity<Length>> {
        let direction = &self.direction;
        let (ax, ay, az) = (direction[0].abs(), direction[1].abs(), direction[2].abs());
        let kz = if ax > ay {
            if ax > az { 0 } else { 2 }
        } else if ay > az {
            1
        } else {
            2
        };
        let (mut kx, mut ky) = ((kz + 1) % 3, (kz + 2) % 3);
        if direction[kz] < 0.0 {
            swap(&mut kx, &mut ky);
        }
        let sx = direction[kx] / direction[kz];
        let sy = direction[ky] / direction[kz];
        let sz = 1.0 / direction[kz];
        let pa = a - &self.origin;
        let pb = b - &self.origin;
        let pc = c - &self.origin;
        let ax = pa[kx] - sx * pa[kz];
        let ay = pa[ky] - sy * pa[kz];
        let bx = pb[kx] - sx * pb[kz];
        let by = pb[ky] - sy * pb[kz];
        let cx = pc[kx] - sx * pc[kz];
        let cy = pc[ky] - sy * pc[kz];
        let u = cx * by - cy * bx;
        let v = ax * cy - ay * cx;
        let w = bx * ay - by * ax;
        let zero = Quantity::<Area>::default();
        if (u < zero || v < zero || w < zero) && (u > zero || v > zero || w > zero) {
            return None;
        }
        let determinant = u + v + w;
        if determinant.abs().value() < ABS_TOL {
            return None;
        }
        let t = (u * sz * pa[kz] + v * sz * pb[kz] + w * sz * pc[kz]) / determinant;
        (t.value() > ABS_TOL).then_some(t)
    }
}
