#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Direction, bvh::ray::Ray},
    math::{Reference, TensorRank1},
    units::{Dimensionless, UnitDiv},
};
use core::array::from_fn;

impl<const D: usize, U> From<(Coordinate<D>, TensorRank1<D, Reference, U>)> for Ray<D>
where
    U: UnitDiv<U, Output = Dimensionless>,
{
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
