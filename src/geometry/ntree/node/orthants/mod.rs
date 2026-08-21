use crate::geometry::ntree::node::{Kind, Node, Orthants, cell::Cell};
use std::array::from_fn;

impl<const D: usize, const M: usize, const N: usize, T, U, V> Node<D, M, N, T, U, V>
where
    T: Cell,
{
    pub fn center(&self) -> [T; D] {
        if self.is_unit() {
            panic!()
        }
        let half = self.length.split();
        from_fn(|axis| self.corner[axis] + half)
    }
}

impl<const D: usize, const M: usize, const N: usize, T, U, V> Node<D, M, N, T, U, V>
where
    T: Cell,
{
    pub fn facets(&self) -> &[Option<U>; M] {
        &self.facets
    }
    pub fn is_leaf(&self) -> bool {
        matches!(self.kind, Kind::Leaf)
    }
    pub fn is_tree(&self) -> bool {
        matches!(self.kind, Kind::Tree(_))
    }
    pub fn is_unit(&self) -> bool {
        self.length == T::ONE
    }
    pub fn orthants(&self) -> Option<&Orthants<N, U>> {
        match &self.kind {
            Kind::Leaf => None,
            Kind::Tree(orthants) => Some(orthants),
        }
    }
}
