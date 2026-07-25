pub(super) mod base;
pub(super) mod from;
pub(super) mod into;

use crate::geometry::{Coordinate, bbox::BoundingBox};

pub struct Primitive<const D: usize> {
    bounding_box: BoundingBox<D>,
    centroid: Coordinate<D>,
    index: usize,
}

pub(super) type Primitives<const D: usize> = Vec<Primitive<D>>;
