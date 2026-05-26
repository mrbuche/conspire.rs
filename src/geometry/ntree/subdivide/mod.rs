use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, Node, split::Split},
};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const M: usize, const N: usize, T> Orthotree<D, M, N, T, usize>
where
    T: AddAssign + Copy + Default + Split,
{
    pub fn subdivide(&mut self, index: usize) -> Result<(), OrthotreeError> {
        if index >= self.nodes.len() {
            return Err(OrthotreeError::IndexOutOfBounds);
        }
        let Kind::Leaf = self.nodes[index].kind else {
            return Err(OrthotreeError::IndexOutOfBounds);
        };
        let corner = self.nodes[index].corner;
        let length = self.nodes[index].length.split();
        let parent_facets = self.nodes[index].facets;
        let first = self.nodes.len();
        for i in 0..N {
            self.nodes.push(Node {
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
                        parent_facets[f]
                    } else {
                        first + (i ^ (1 << axis))
                    }
                }),
                kind: Kind::Leaf,
            });
        }
        self.nodes[index].kind = Kind::Tree {
            orthants: from_fn(|i| first + i),
        };
        Ok(())
    }
}
