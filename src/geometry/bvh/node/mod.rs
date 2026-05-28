pub mod from;

use crate::geometry::bbox::BoundingBox;

pub struct Node<const D: usize> {
    bounding_box: BoundingBox<D>,
    kind: NodeKind,
}

pub enum NodeKind {
    Leaf { start: usize, end: usize },
    Tree { left: usize, right: usize },
}

pub type Nodes<const D: usize> = Vec<Node<D>>;
