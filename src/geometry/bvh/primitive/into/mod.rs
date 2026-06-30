#[cfg(test)]
mod test;

use crate::geometry::{
    bbox::{BoundingBox, Unite},
    bvh::primitive::Primitive,
};

impl<const D: usize> From<&[Primitive<D>]> for BoundingBox<D> {
    fn from(primitives: &[Primitive<D>]) -> Self {
        primitives
            .iter()
            .skip(1)
            .fold(primitives[0].bounding_box.clone(), |bbox, primitive| {
                bbox.unite(&primitive.bounding_box)
            })
    }
}
