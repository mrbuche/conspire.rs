mod base;
mod from;
mod node;
mod primitive;

use crate::geometry::bvh::node::Nodes;

pub struct BoundingVolumeHierarchy<const D: usize> {
    items: Vec<usize>,
    nodes: Nodes<D>,
}
