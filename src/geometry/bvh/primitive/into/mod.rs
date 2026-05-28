#[cfg(test)]
mod test;

use crate::geometry::{BoundingBox, BoundingBoxUnite, bvh::primitive::Primitive};

impl<const D: usize, T> From<&[Primitive<D, T>]> for BoundingBox<D> {
    fn from(primitives: &[Primitive<D, T>]) -> Self {
        primitives
            .iter()
            .skip(1)
            .fold(primitives[0].bounding_box.clone(), |bbox, primitive| {
                bbox.unite(&primitive.bounding_box)
            })
    }
}
