pub mod base;
pub mod index;
pub mod node;

use crate::geometry::ntree::node::Nodes;

pub struct Orthotree<const D: usize, const N: usize, T, U> {
    nodes: Nodes<D, N, T, U>,
}

pub type Quadtree<T, U> = Orthotree<2, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 8, T, U>;

// move to error Enum? for display impls etc.?
pub enum OrthotreeError {
    SubdivideTree,
}
