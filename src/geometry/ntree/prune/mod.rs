use crate::geometry::ntree::Orthotree;
use crate::geometry::ntree::node::cell::Cell;

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
where
    T: Cell,
{
    pub fn prune(&mut self) {
        self.retain(|node| node.is_leaf());
    }
}
