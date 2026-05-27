use crate::geometry::ntree::{Orthotree, node::Nodes};
use std::ops::{Deref, DerefMut};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U> Deref
    for Orthotree<D, L, M, N, T, U>
{
    type Target = Nodes<D, M, N, T, U>;
    fn deref(&self) -> &Self::Target {
        &self.nodes
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U> DerefMut
    for Orthotree<D, L, M, N, T, U>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.nodes
    }
}
