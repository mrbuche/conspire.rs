#[cfg(test)]
mod test;

use crate::geometry::{
    bbox::BoundingBox,
    bvh::node::{Node, NodeKind},
};

impl<const D: usize> From<(BoundingBox<D>, NodeKind)> for Node<D> {
    fn from((bounding_box, kind): (BoundingBox<D>, NodeKind)) -> Self {
        Self { bounding_box, kind }
    }
}

impl<const D: usize> From<(&BoundingBox<D>, NodeKind)> for Node<D> {
    fn from((bounding_box, kind): (&BoundingBox<D>, NodeKind)) -> Self {
        Self {
            bounding_box: bounding_box.clone(),
            kind,
        }
    }
}
