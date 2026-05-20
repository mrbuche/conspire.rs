pub mod from;

use crate::geometry::{BoundingBox, Coordinate};

pub struct Item<const D: usize, const I: usize, T> {
    bounding_box: BoundingBox<D, I>,
    centroid: Coordinate<D, I>,
    index: T,
}

pub type Items<const D: usize, const I: usize, T> = Vec<Item<D, I, T>>;
