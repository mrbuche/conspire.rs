pub mod from;
pub mod union;

use crate::geometry::Coordinate;

#[derive(Debug, PartialEq)]
pub struct BoundingBox<const D: usize, const I: usize> {
    minimum: Coordinate<D, I>,
    maximum: Coordinate<D, I>,
}

pub trait Union<T> {
    type Output;
    fn union(self, other: T) -> Self::Output;
}
