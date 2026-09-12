use crate::geometry::ntree::node::slot::Slot;
use crate::geometry::ntree::{Orthotree, node::cell::Cell};
use std::array::from_fn;

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Cell,
    U: Slot,
    V: Copy,
{
    pub(super) fn pair_regular(&mut self) -> Result<bool, &'static str> {
        let mut index = 0;
        let mut paired = true;
        self.pairing_vertices.clear();
        while index < self.len() {
            if let Some(nodes) = self.nodes[index].orthants() {
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
                        self.subdivide(node.slot())?;
                    }
                } else if any_tree {
                    let half = self.nodes[index].length.split();
                    let corner = self.nodes[index].corner;
                    let center = from_fn(|axis| (corner[axis] + half).cells());
                    self.pairing_vertices.insert((center, half.cells()));
                }
            }
            index += 1;
        }
        Ok(paired)
    }
}
