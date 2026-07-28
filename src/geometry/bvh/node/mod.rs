pub(super) mod from;

use crate::geometry::bbox::BoundingBox;

pub(super) struct Node<const D: usize> {
    bounding_box: BoundingBox<D>,
    kind: NodeKind,
}

impl<const D: usize> Node<D> {
    pub(super) fn bounding_box(&self) -> &BoundingBox<D> {
        &self.bounding_box
    }
    pub(super) fn kind(&self) -> &NodeKind {
        &self.kind
    }
}

pub(super) enum NodeKind {
    Leaf { start: usize, end: usize },
    Tree { left: usize, right: usize },
}

pub(super) type Nodes<const D: usize> = Vec<Node<D>>;
