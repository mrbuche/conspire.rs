#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::primitive::{Primitive, Primitives},
    mesh::Mesh,
};

impl<const D: usize> From<&Mesh<D>> for Primitives<D> {
    fn from(mesh: &Mesh<D>) -> Self {
        mesh.bounding_boxes_and_centroids()
            .enumerate()
            .map(|(primitive, (bounding_box, centroid))| Primitive {
                bounding_box,
                centroid,
                index: primitive,
            })
            .collect()
    }
}
