use crate::geometry::ntree::{
    Orthotree,
    node::{Kind, Node},
};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    pub fn leaves<'a>(&self, node: &'a Node<D, M, N, T, U>) -> Option<&'a [U; N]> {
        match &node.kind {
            Kind::Leaf => None,
            Kind::Tree(orthants) => {
                if orthants.iter().any(|&orthant| self[orthant].is_tree()) {
                    None
                } else {
                    Some(orthants)
                }
            }
        }
    }
    pub fn prune(&mut self) {
        self.retain(|node| node.is_leaf());
    }
}
