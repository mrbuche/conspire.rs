pub mod base;
pub mod from;
pub mod into;

use crate::geometry::{BoundingBox, Coordinate};

pub struct Primitive<const D: usize, T> {
    bounding_box: BoundingBox<D>,
    centroid: Coordinate<D>,
    index: T,
}

pub type Primitives<const D: usize, T> = Vec<Primitive<D, T>>;
