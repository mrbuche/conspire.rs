pub mod from;

use crate::geometry::bbox::BoundingBox;

pub struct Node<const D: usize> {
    bounding_box: BoundingBox<D>,
    kind: NodeKind,
}

impl<const D: usize> Node<D> {
    pub fn bounding_box(&self) -> &BoundingBox<D> {
        &self.bounding_box
    }
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }
}

pub enum NodeKind {
    Leaf { start: usize, end: usize },
    Tree { left: usize, right: usize },
}

pub type Nodes<const D: usize> = Vec<Node<D>>;
