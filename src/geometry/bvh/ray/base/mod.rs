#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, bbox::BoundingBox, bvh::ray::Ray},
    math::{Scalar, Tensor},
};

impl<const D: usize> Ray<D> {
    pub fn new(origin: Coordinate<D>, direction: Coordinate<D>) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }
    pub fn origin(&self) -> &Coordinate<D> {
        &self.origin
    }
    pub fn direction(&self) -> &Coordinate<D> {
        &self.direction
    }
    pub fn intersects(&self, bounding_box: &BoundingBox<D>) -> Option<Scalar> {
        let mut t_min: Scalar = 0.0; // can return third case of inside box
        let mut t_max: Scalar = Scalar::INFINITY;
        for axis in 0..D {
            let inverse_direction = 1.0 / self.direction[axis];
            let mut t_near = (bounding_box.minimum()[axis] - self.origin[axis]) * inverse_direction;
            let mut t_far = (bounding_box.maximum()[axis] - self.origin[axis]) * inverse_direction;
            if inverse_direction < 0.0 {
                std::mem::swap(&mut t_near, &mut t_far);
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