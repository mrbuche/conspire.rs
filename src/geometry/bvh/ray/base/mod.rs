#[cfg(test)]
mod test;

use std::mem::swap;

use crate::{
    ABS_TOL,
    geometry::{Coordinate, bbox::BoundingBox, bvh::ray::Ray},
    math::{CrossProduct, Scalar},
};

impl<const D: usize> Ray<D> {
    pub fn origin(&self) -> &Coordinate<D> {
        &self.origin
    }
    pub fn direction(&self) -> &Coordinate<D> {
        &self.direction
    }
    pub fn intersects(&self, bounding_box: &BoundingBox<D>) -> Option<Scalar> {
        let mut t_min: Scalar = 0.0; // can return third case of inside box using custom enum
        let mut t_max: Scalar = Scalar::INFINITY;
        for axis in 0..D {
            let inverse_direction = 1.0 / self.direction[axis];
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
    pub fn intersects_triangle(
        &self,
        a: &Coordinate<3>,
        b: &Coordinate<3>,
        c: &Coordinate<3>,
    ) -> Option<Scalar> {
        let edge_1 = b - a;
        let edge_2 = c - a;
        let p = self.direction.cross(&edge_2);
        let determinant = &edge_1 * &p;
        if determinant.abs() < ABS_TOL {
            return None;
        }
        let inverse_determinant = 1.0 / determinant;
        let s = &self.origin - a;
        let u = inverse_determinant * (&s * &p);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(&edge_1);
        let v = inverse_determinant * (&self.direction * &q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = inverse_determinant * (&edge_2 * &q);
        (t > ABS_TOL).then_some(t)
    }
}
