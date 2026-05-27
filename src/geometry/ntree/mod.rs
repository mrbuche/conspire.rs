pub mod balance;
pub mod deref;
pub mod dual;
pub mod error;
pub mod from;
pub mod index;
pub mod into;
pub mod leaves;
pub mod node;
pub mod pair;
pub mod prune;
pub mod subdivide;

use crate::geometry::ntree::node::Nodes;

pub struct Orthotree<const D: usize, const L: usize, const M: usize, const N: usize, T, U> {
    nodes: Nodes<D, M, N, T, U>,
}

pub type BinaryTree<T, U> = Orthotree<1, 1, 2, 2, T, U>;
pub type Quadtree<T, U> = Orthotree<2, 2, 4, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 4, 6, 8, T, U>;
pub type Hexadecatree<T, U> = Orthotree<4, 8, 8, 16, T, U>;
