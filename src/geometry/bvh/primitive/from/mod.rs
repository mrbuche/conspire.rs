#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::primitive::{Primitive, Primitives},
    mesh::Mesh,
};

impl<const D: usize, const M: usize, T, U, V> From<&Mesh<D, M, T>> for Primitives<D, V>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(mesh: &Mesh<D, M, T>) -> Self {
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
