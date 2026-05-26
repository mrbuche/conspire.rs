use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, Node, sentinel::Sentinel, split::Split},
};
use std::{array::from_fn, ops::AddAssign};

#[derive(Clone, Copy)]
pub enum Pairing {
    Regular,
    None,
}

impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
where
    T: AddAssign + Copy + Default + PartialEq + Split,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn subdivide(&mut self, index: usize, pairing: Pairing) -> Result<(), OrthotreeError> {
        if index >= self.nodes.len() {
            return Err(OrthotreeError::IndexOutOfBounds);
        }
        let Kind::Leaf = self.nodes[index].kind else {
            return Err(OrthotreeError::IndexOutOfBounds);
        };
        match pairing {
            Pairing::None => self.subdivide_node(index),
            Pairing::Regular => {
                let length = self.nodes[index].length;
                let mut siblings = Vec::with_capacity(N);
                siblings.push(index);
                let mut i = 0;
                while i < siblings.len() {
                    let current = siblings[i];
                    for &neighbor in &self.nodes[current].facets {
                        if neighbor != U::MAX
                            && self.nodes[neighbor.into()].length == length
                            && !siblings.contains(&neighbor.into())
                        {
                            siblings.push(neighbor.into());
                        }
                    }
                    i += 1;
                }
                for sibling in siblings {
                    if self.nodes[sibling].is_leaf() {
                        self.subdivide_node(sibling)?;
                    }
                }
                Ok(())
            }
        }
    }
    fn subdivide_node(&mut self, index: usize) -> Result<(), OrthotreeError> {
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
                        U::from(first + (i ^ (1 << axis)))
                    }
                }),
                kind: Kind::Leaf,
            });
        }
        self.nodes[index].kind = Kind::Tree(from_fn(|i| U::from(first + i)));
        Ok(())
    }
}
