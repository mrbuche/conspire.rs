use crate::geometry::ntree::Orthotree;

impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U> {
    pub fn prune(&mut self) {
        self.nodes.retain(|node| node.is_leaf());
    }
}
