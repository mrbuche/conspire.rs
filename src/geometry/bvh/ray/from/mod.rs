#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, bvh::ray::Ray},
    math::Tensor,
};
use std::array::from_fn;

impl<const D: usize> From<(Coordinate<D>, Coordinate<D>)> for Ray<D> {
    fn from((origin, direction): (Coordinate<D>, Coordinate<D>)) -> Self {
        let direction = direction.normalized();
        let inverse_direction = Coordinate::from(from_fn(|i| 1.0 / direction[i]));
        Self {
            origin,
            direction,
            inverse_direction,
        }
    }
}
