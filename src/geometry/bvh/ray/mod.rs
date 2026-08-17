mod base;
mod from;

use crate::geometry::{Coordinate, Direction};

#[derive(Clone, Debug, PartialEq)]
pub struct Ray<const D: usize> {
    origin: Coordinate<D>,
    direction: Direction<D>,
    inverse_direction: Direction<D>,
}
