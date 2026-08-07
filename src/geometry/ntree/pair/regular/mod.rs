use crate::geometry::ntree::{Orthotree, node::split::Split};
use std::{array::from_fn, ops::Add};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    pub(super) fn pair_regular(&mut self) -> Result<bool, &'static str> {
        let mut index = 0;
        let mut paired = true;
        self.pairing_vertices.clear();
        while index < self.len() {
            if let Some(nodes) = self[index.into()].orthants() {
                let mut any_leaf = false;
                let mut any_tree = false;
                let mut leaves = Vec::with_capacity(N);
                for &node in nodes.iter() {
                    if self[node].is_leaf() {
                        any_leaf = true;
                        leaves.push(node);
                    } else if self[node].is_tree() {
                        any_tree = true;
                    }
                }
                if any_tree && any_leaf {
                    for node in leaves {
                        paired = false;
                        self.subdivide(node)?;
                    }
                } else if any_tree {
                    let half = self[index.into()].length.split();
                    let corner = self[index.into()].corner;
                    let center = from_fn(|axis| (corner[axis] + half).into());
                    self.pairing_vertices.insert((center, half.into()));
                }
            }
            index += 1;
        }
        Ok(paired)
    }
}
