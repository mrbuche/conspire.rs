#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, bbox::BoundingBox, bvh::primitive::Primitive};

impl<const D: usize> Primitive<D> {
    pub fn bounding_box(&self) -> &BoundingBox<D> {
        &self.bounding_box
    }
    pub fn centroid(&self) -> &Coordinate<D> {
        &self.centroid
    }
    pub fn index(&self) -> usize {
        self.index
    }
}
