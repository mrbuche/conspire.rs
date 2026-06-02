mod base;

use crate::geometry::Coordinate;

pub struct Ray<const D: usize> {
    origin: Coordinate<D>,
    direction: Coordinate<D>,
}
