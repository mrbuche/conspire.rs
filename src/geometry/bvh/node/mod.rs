pub mod from;

use crate::geometry::bbox::BoundingBox;

pub enum NodeKind {
    Leaf { start: usize, end: usize },
    Tree { left: usize, right: usize },
}

pub struct Node<const D: usize, const I: usize> {
    bounding_box: BoundingBox<D, I>,
    kind: NodeKind,
}

pub type Nodes<const D: usize, const I: usize> = Vec<Node<D, I>>;
