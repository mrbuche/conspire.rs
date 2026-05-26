use crate::geometry::ntree::node::{Kind, Node, Orthants};

impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U> {
    pub fn is_leaf(&self) -> bool {
        matches!(self.kind, Kind::Leaf)
    }
    pub fn is_tree(&self) -> bool {
        matches!(self.kind, Kind::Tree(_))
    }
    pub fn orthants(&self) -> &Orthants<N, U> {
        match &self.kind {
            Kind::Leaf => panic!(),
            Kind::Tree(orthants) => orthants,
        }
    }
}
