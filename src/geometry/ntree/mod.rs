mod balance;
mod defeature;
mod deref;
mod dual;
mod from;
mod index;
mod into;
mod leaves;
mod node;
mod pair;
mod prune;
mod read;
mod rescale;
mod subdivide;
mod write;

pub use crate::geometry::ntree::{
    balance::{Balance, Balancing},
    dual::Dualization,
    from::CurvatureSizing,
    node::Nodes,
    pair::Pairing,
    read::Input,
    rescale::Rescaling,
    write::Output,
};

pub struct Orthotree<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V = ()> {
    balanced: Balancing,
    nodes: Nodes<D, M, N, T, U, V>,
    paired: Pairing,
    rescale: Rescaling<D>,
}

pub type BinaryTree<T, U, V = ()> = Orthotree<1, 1, 2, 2, T, U, V>;
pub type Quadtree<T, U, V = ()> = Orthotree<2, 2, 4, 4, T, U, V>;
pub type Octree<T, U, V = ()> = Orthotree<3, 4, 6, 8, T, U, V>;
pub type Hexadecatree<T, U, V = ()> = Orthotree<4, 8, 8, 16, T, U, V>;
