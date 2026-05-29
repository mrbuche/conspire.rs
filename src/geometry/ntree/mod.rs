mod balance;
mod deref;
mod dual;
mod error;
mod from;
mod index;
mod into;
mod leaves;
mod node;
mod pair;
mod prune;
mod subdivide;

pub use crate::geometry::ntree::{
    balance::{Balance, Balancing},
    dual::Dualization,
    node::Nodes,
    pair::Pairing,
};

pub struct Orthotree<const D: usize, const L: usize, const M: usize, const N: usize, T, U> {
    balanced: Balancing,
    nodes: Nodes<D, M, N, T, U>,
    paired: Pairing,
}

pub type BinaryTree<T, U> = Orthotree<1, 1, 2, 2, T, U>;
pub type Quadtree<T, U> = Orthotree<2, 2, 4, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 4, 6, 8, T, U>;
pub type Hexadecatree<T, U> = Orthotree<4, 8, 8, 16, T, U>;
