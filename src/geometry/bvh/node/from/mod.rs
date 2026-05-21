#[cfg(test)]
mod test;

use crate::geometry::{
    bbox::BoundingBox,
    bvh::node::{Node, NodeKind},
};

impl<const D: usize, const I: usize> From<(BoundingBox<D, I>, NodeKind)> for Node<D, I> {
    fn from((bounding_box, kind): (BoundingBox<D, I>, NodeKind)) -> Self {
        Self { bounding_box, kind }
    }
}

impl<const D: usize, const I: usize> From<(&BoundingBox<D, I>, NodeKind)> for Node<D, I> {
    fn from((bounding_box, kind): (&BoundingBox<D, I>, NodeKind)) -> Self {
        Self {
            bounding_box: bounding_box.clone(),
            kind,
        }
    }
}
