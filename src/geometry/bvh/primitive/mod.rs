pub mod base;
pub mod from;

use crate::geometry::{BoundingBox, Coordinate};

pub struct Primitive<const D: usize, const I: usize, T> {
    bounding_box: BoundingBox<D, I>,
    centroid: Coordinate<D, I>,
    index: T,
}

pub type Primitives<const D: usize, const I: usize, T> = Vec<Primitive<D, I, T>>;
