pub mod base;
pub mod from;
pub mod into;

use crate::geometry::{Coordinate, bbox::BoundingBox};

pub struct Primitive<const D: usize> {
    bounding_box: BoundingBox<D>,
    centroid: Coordinate<D>,
    index: usize,
}

pub type Primitives<const D: usize> = Vec<Primitive<D>>;
