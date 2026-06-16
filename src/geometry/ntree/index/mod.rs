use crate::geometry::ntree::{Orthotree, node::Node};
use std::ops::{Index, IndexMut};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> Index<U>
    for Orthotree<D, L, M, N, T, U, V>
where
    U: Into<usize>,
{
    type Output = Node<D, M, N, T, U, V>;
    fn index(&self, idx: U) -> &Self::Output {
        &self.nodes[idx.into()]
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> IndexMut<U>
    for Orthotree<D, L, M, N, T, U, V>
where
    U: Into<usize>,
{
    fn index_mut(&mut self, idx: U) -> &mut Self::Output {
        &mut self.nodes[idx.into()]
    }
}
