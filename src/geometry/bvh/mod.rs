use crate::geometry::{BoundingBox, BoundingBoxUnion};

pub struct Node<const D: usize, const I: usize> {
    bounding_box: BoundingBox<D, I>,
}

pub struct BoundingVolumeHierarchy<const D: usize, const I: usize> {
    nodes: Vec<Node<D, I>>,
    items: Vec<usize>,
}
