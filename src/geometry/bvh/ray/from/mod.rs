#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Direction, bvh::ray::Ray},
    math::{Reference, TensorRank1},
};
use std::array::from_fn;

/// A ray is cast along whatever vector is handed over, an edge span as readily
/// as a heading, since normalizing spends whatever it was measured in.
impl<const D: usize, U> From<(Coordinate<D>, TensorRank1<D, Reference, U>)> for Ray<D> {
    fn from((origin, direction): (Coordinate<D>, TensorRank1<D, Reference, U>)) -> Self {
        let direction = direction.normalized();
        let inverse_direction = Direction::from(from_fn(|i| 1.0 / direction[i]));
        Self {
            origin,
            direction,
            inverse_direction,
        }
    }
}
