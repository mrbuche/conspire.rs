use crate::geometry::ntree::{Orthotree, leaf::Leaf};
use std::ops::{Index, IndexMut};

impl<const D: usize, const N: usize, T, U> Index<usize> for Orthotree<D, N, T, U> {
    type Output = Leaf<D, T, U>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.leaves[idx]
    }
}

impl<const D: usize, const N: usize, T, U> IndexMut<usize> for Orthotree<D, N, T, U> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.leaves[idx]
    }
}
