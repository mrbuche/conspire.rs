mod balance;
mod defeature;
mod deref;
mod from;
mod index;
mod into;
mod leaves;
pub(crate) mod node;
mod pair;
mod prune;
mod read;
pub(crate) mod rescale;
pub(crate) mod sizing;
pub(crate) mod subdivide;
mod write;

pub use crate::geometry::ntree::{
    balance::{Balance, Balancing},
    node::Nodes,
    pair::Pairing,
    read::Input,
    rescale::Rescaling,
    sizing::{Sizing, curvature::CurvatureSizing},
    write::Output,
};
use crate::math::FxHashSet;

pub struct Orthotree<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V = ()> {
    pub(crate) balanced: Balancing,
    pub(crate) nodes: Nodes<D, M, N, T, U, V>,
    pub(crate) paired: Pairing,
    // Centers of the 2x2 clusters of refined cells the pairing settled on, each with the
    // length of the cells it groups. Recorded by every pairing scheme; read by the dual.
    pub(crate) pairing_vertices: FxHashSet<([usize; D], usize)>,
    // The node count last time `Pairing::Generalized` found the tree already fully paired.
    // `subdivide` is the only way the tree structure ever changes, and always grows `nodes`,
    // so an unchanged length here is a cheap, exact proof that nothing has changed anywhere
    // (not just at any one level) since that pairing was computed - not a heuristic.
    pub(crate) pairing_stable_len: Option<usize>,
    pub(crate) rescale: Rescaling<D>,
}

pub type BinaryTree<T, U, V = ()> = Orthotree<1, 1, 2, 2, T, U, V>;
pub type Quadtree<T, U, V = ()> = Orthotree<2, 2, 4, 4, T, U, V>;
pub type Octree<T, U, V = ()> = Orthotree<3, 4, 6, 8, T, U, V>;
pub type Hexadecatree<T, U, V = ()> = Orthotree<4, 8, 8, 16, T, U, V>;
