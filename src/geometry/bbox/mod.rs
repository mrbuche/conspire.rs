pub mod base;
pub mod from;
pub mod unite;

use crate::geometry::Coordinate;

#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<const D: usize> {
    minimum: Coordinate<D>,
    maximum: Coordinate<D>,
}

pub type BoundingBoxes<const D: usize> = Vec<BoundingBox<D>>;

pub trait Unite<T> {
    type Output;
    fn unite(self, other: T) -> Self::Output;
}
