pub mod base;
pub mod kind;
pub mod orthants;

use crate::geometry::ntree::node::kind::Kind;

pub struct Node<const D: usize, const N: usize, T, U> {
    corner: [T; D],
    length: T,
    kind: Kind<N, U>,
}

pub type Nodes<const D: usize, const N: usize, T, U> = Vec<Node<D, N, T, U>>;
