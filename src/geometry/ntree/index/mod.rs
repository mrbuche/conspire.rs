use crate::geometry::ntree::{Orthotree, node::Node};
use std::ops::{Index, IndexMut};

impl<const D: usize, const M: usize, const N: usize, T, U> Index<U> for Orthotree<D, M, N, T, U>
where
    U: Into<usize>,
{
    type Output = Node<D, M, N, T, U>;
    fn index(&self, idx: U) -> &Self::Output {
        &self.nodes[idx.into()]
    }
}

impl<const D: usize, const M: usize, const N: usize, T, U> IndexMut<U> for Orthotree<D, M, N, T, U>
where
    U: Into<usize>,
{
    fn index_mut(&mut self, idx: U) -> &mut Self::Output {
        &mut self.nodes[idx.into()]
    }
}
