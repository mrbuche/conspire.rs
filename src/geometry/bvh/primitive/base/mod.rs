#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, bbox::BoundingBox, bvh::primitive::Primitive};

impl<const D: usize> Primitive<D> {
    pub(in crate::geometry::bvh) fn new(
        bounding_box: BoundingBox<D>,
        centroid: Coordinate<D>,
        index: usize,
    ) -> Self {
        Self {
            bounding_box,
            centroid,
            index,
        }
    }
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
