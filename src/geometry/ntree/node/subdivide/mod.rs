use crate::geometry::ntree::{
    error::OrthotreeError,
    node::{Kind, Node, Orthants, split::Split},
};

impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U>
where
    T: Copy + Split,
{
    pub fn subdivide(&self, indices: [U; N]) -> Result<[Self; N], OrthotreeError> {
        match self.kind {
            Kind::Leaf => {
                let length = self.length.split();
                todo!()
            }
            Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideLeaf),
        }
    }
}
