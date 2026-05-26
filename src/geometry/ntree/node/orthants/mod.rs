use crate::geometry::ntree::node::{Kind, Node, Orthants};

impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U>
where
    T: Copy + Into<usize>,
{
    pub fn is_leaf(&self) -> bool {
        matches!(self.kind, Kind::Leaf)
    }
    pub fn is_tree(&self) -> bool {
        matches!(self.kind, Kind::Tree(_))
    }
    pub fn orthants(&self) -> Option<&Orthants<N, U>> {
        match &self.kind {
            Kind::Leaf => None,
            Kind::Tree(orthants) => Some(orthants),
        }
    }
    //
    // temporary
    //
    pub fn get_faces(&self) -> &[Option<U>; M] {
        &self.facets
    }
    pub fn is_voxel(&self) -> bool {
        self.length.into() == 1
    }
}
