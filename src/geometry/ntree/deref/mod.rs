use crate::geometry::ntree::{Orthotree, node::Nodes};
use std::ops::{Deref, DerefMut};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> Deref
    for Orthotree<D, L, M, N, T, U, V>
{
    type Target = Nodes<D, M, N, T, U, V>;
    fn deref(&self) -> &Self::Target {
        &self.nodes
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> DerefMut
    for Orthotree<D, L, M, N, T, U, V>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.nodes
    }
}
