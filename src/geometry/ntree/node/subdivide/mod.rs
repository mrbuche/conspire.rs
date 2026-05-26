use crate::geometry::ntree::{
    error::OrthotreeError,
    node::{Kind, Node, Orthants, split::Split},
};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U>
where
    T: AddAssign + Copy + Split,
    U: Copy,
{
    pub fn subdivide(&self, indices: [U; N]) -> Result<[Self; N], OrthotreeError> {
        match self.kind {
            Kind::Leaf => {
                let length = self.length.split();
                let corner = self.corner;
                let facets = self.facets;
                Ok(from_fn(|i| Node {
                    corner: from_fn(|ax| {
                        if (i >> ax) & 1 == 1 {
                            let mut c = corner[ax];
                            c += length;
                            c
                        } else {
                            corner[ax]
                        }
                    }),
                    length,
                    facets: from_fn(|f| {
                        let axis = f / 2;
                        let sign = f % 2;
                        if (i >> axis) & 1 == sign {
                            facets[f]
                        } else {
                            indices[i ^ (1 << axis)]
                        }
                    }),
                    kind: Kind::Leaf,
                }))
            }
            Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideLeaf),
        }
    }
}
