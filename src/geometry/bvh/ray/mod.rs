mod base;
mod from;

use crate::geometry::Coordinate;

#[derive(Clone, Debug, PartialEq)]
pub struct Ray<const D: usize> {
    origin: Coordinate<D>,
    direction: Coordinate<D>,
}
