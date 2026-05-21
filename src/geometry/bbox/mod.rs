pub mod base;
pub mod from;
pub mod unite;

use crate::geometry::Coordinate;

#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<const D: usize, const I: usize> {
    minimum: Coordinate<D, I>,
    maximum: Coordinate<D, I>,
}

pub type BoundingBoxes<const D: usize, const I: usize> = Vec<BoundingBox<D, I>>;

pub trait Unite<T> {
    type Output;
    fn unite(self, other: T) -> Self::Output;
}
