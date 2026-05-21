#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, bbox::BoundingBox, bvh::primitive::Primitive};

impl<const D: usize, const I: usize, T> Primitive<D, I, T>
where
    T: Copy,
{
    pub fn bounding_box(&self) -> &BoundingBox<D, I> {
        &self.bounding_box
    }
    pub fn centroid(&self) -> &Coordinate<D, I> {
        &self.centroid
    }
    pub fn index(&self) -> T {
        self.index
    }
}
