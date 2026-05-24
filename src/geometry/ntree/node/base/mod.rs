use crate::geometry::ntree::{
    error::OrthotreeError,
    node::{Kind, Node, orthants::Orthants},
};
use std::{array::from_fn, ops::AddAssign};

pub trait Split {
    fn split(self) -> Self;
}

impl Split for usize {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for u16 {
    fn split(self) -> Self {
        self / 2
    }
}

impl<const D: usize, const N: usize, T, U> Node<D, N, T, U>
where
    T: AddAssign + Copy + Split,
{
    fn is_leaf(&self) -> bool {
        match self.kind {
            Kind::Leaf => true,
            Kind::Tree(_) => false,
        }
    }
    fn is_tree(&self) -> bool {
        match self.kind {
            Kind::Leaf => false,
            Kind::Tree(_) => true,
        }
    }
    fn orthants(&self) -> Option<&Orthants<N, U>> {
        match &self.kind {
            Kind::Leaf => None,
            Kind::Tree(orthants) => Some(orthants),
        }
    }
    pub fn try_subdivide(
        &mut self,
        indices: [U; N],
    ) -> Result<[Node<D, N, T, U>; N], OrthotreeError> {
        match self.kind {
            Kind::Leaf => {
                self.kind = indices.into();
                let length = self.length.split();
                let children = from_fn(|i| {
                    let mut corner = self.corner;
                    corner.iter_mut().enumerate().for_each(|(axis, coord)| {
                        if (i >> axis) & 1 == 1 {
                            *coord += length;
                        }
                    });
                    Node {
                        corner,
                        length,
                        kind: Kind::Leaf,
                    }
                });
                Ok(children)
            }
            Kind::Tree(_) => Err(OrthotreeError::SubdivideTree),
        }
    }
}
