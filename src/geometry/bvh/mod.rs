pub mod base;
pub mod from;
pub mod node;
pub mod primitive;

use crate::geometry::bvh::node::Nodes;

pub struct BoundingVolumeHierarchy<const D: usize, const I: usize, T> {
    items: Vec<T>,
    nodes: Nodes<D, I>,
}
