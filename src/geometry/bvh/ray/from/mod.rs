#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, bvh::ray::Ray},
    math::Tensor,
};

impl<const D: usize> From<(Coordinate<D>, Coordinate<D>)> for Ray<D> {
    fn from((origin, direction): (Coordinate<D>, Coordinate<D>)) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }
}
