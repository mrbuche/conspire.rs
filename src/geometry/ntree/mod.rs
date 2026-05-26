// pub mod balance;
pub mod error;
// pub mod from;
// pub mod into;
pub mod node;
pub mod subdivide;

use crate::geometry::ntree::node::Nodes;

pub struct Orthotree<const D: usize, const M: usize, const N: usize, T, U> {
    nodes: Nodes<D, M, N, T, U>,
}

pub type BinaryTree<T, U> = Orthotree<1, 2, 2, T, U>;
pub type Quadtree<T, U> = Orthotree<2, 4, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 6, 8, T, U>;
pub type Hexadecatree<T, U> = Orthotree<4, 8, 16, T, U>;
