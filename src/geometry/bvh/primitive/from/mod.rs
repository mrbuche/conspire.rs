#[cfg(test)]
mod test;

use crate::geometry::{
    BoundingBox, BoundingBoxUnite,
    bvh::primitive::{Primitive, Primitives},
    mesh::Mesh,
};
use std::iter::ExactSizeIterator;

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

impl<const D: usize, const I: usize, const M: usize, T, U, V> From<&Mesh<D, I, M, T>>
    for Primitives<D, I, V>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    for<'a> <&'a U as IntoIterator>::IntoIter: ExactSizeIterator,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(mesh: &Mesh<D, I, M, T>) -> Self {
        mesh.bounding_boxes_and_centroids()
            .enumerate()
            .map(|(primitive, (bounding_box, centroid))| Primitive {
                bounding_box,
                centroid,
                index: primitive.into(),
            })
            .collect()
    }
}
