#[cfg(test)]
mod test;

use crate::geometry::{BoundingBox, BoundingBoxUnite, bvh::primitive::Primitive};

impl<const D: usize, const I: usize, T> From<&[Primitive<D, I, T>]> for BoundingBox<D, I> {
    fn from(primitives: &[Primitive<D, I, T>]) -> Self {
        primitives
            .iter()
            .skip(1)
            .fold(primitives[0].bounding_box.clone(), |bbox, primitive| {
                bbox.unite(&primitive.bounding_box)
            })
    }
}
