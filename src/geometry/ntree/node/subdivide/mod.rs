use crate::geometry::ntree::node::{Kind, Node, split::Split};
use std::{array::from_fn, ops::Add};

impl<const D: usize, const M: usize, const N: usize, T, U, V> Node<D, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split,
    U: Copy,
    V: Copy,
{
    pub fn subdivide(&self, indices: [U; N]) -> Result<[Self; N], &'static str> {
        match self.kind {
            Kind::Leaf => {
                let corner = self.corner;
                let length = self.length.split();
                Ok(from_fn(|i| Node {
                    corner: from_fn(|axis| {
                        if (i >> axis) & 1 == 1 {
                            corner[axis] + length
                        } else {
                            corner[axis]
                        }
                    }),
                    length,
                    facets: from_fn(|f| {
                        let axis = f / 2;
                        if (i >> axis) & 1 == f % 2 {
                            None
                        } else {
                            Some(indices[i ^ (1 << axis)])
                        }
                    }),
                    kind: Kind::Leaf,
                    value: self.value,
                }))
            }
            Kind::Tree(_) => Err("cannot subdivide a tree"),
        }
    }
}
