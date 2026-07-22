mod ilp;
#[cfg(test)]
mod test;

use crate::geometry::ntree::{Orthotree, node::split::Split};
use ilp::Instance;
use std::{collections::BTreeSet, ops::Add};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    pub(super) fn pair_generalized(&mut self) -> Result<bool, &'static str> {
        let lengths: BTreeSet<usize> = (0..self.len())
            .map(|index| self[U::from(index)].length.into())
            .collect();
        let lengths: Vec<usize> = lengths.into_iter().collect();
        let mut paired = true;
        for window in lengths.windows(2) {
            let (fine, coarse) = (window[0], window[1]);
            if !self.pair_level(coarse, fine)? {
                paired = false;
            }
        }
        Ok(paired)
    }
    fn pair_level(&mut self, coarse: usize, fine: usize) -> Result<bool, &'static str> {
        if coarse != 2 * fine {
            return Err(
                "generalized pairing requires adjacent levels to differ by a factor of two",
            );
        }
        let coarse_nodes: Vec<(U, [i32; D], bool)> = (0..self.len())
            .filter(|&index| self[U::from(index)].length.into() == coarse)
            .map(|index| {
                let idx = U::from(index);
                let required = self[idx].is_tree();
                let corner = self[idx]
                    .corner
                    .map(|coordinate| (coordinate.into() / coarse) as i32);
                (idx, corner, required)
            })
            .collect();
        let instance = Instance::new(
            coarse_nodes
                .iter()
                .map(|&(_, corner, required)| (corner, required))
                .collect(),
        );
        let (assignment, _) = instance.solve();
        debug_assert!(instance.feasible(&assignment));
        let mut paired = true;
        for (index, corner, required) in coarse_nodes {
            if required {
                continue;
            }
            let split = (0..1usize << D).any(|bits| {
                let mut vertex = corner;
                for (axis, coordinate) in vertex.iter_mut().enumerate() {
                    *coordinate += ((bits >> axis) & 1) as i32;
                }
                assignment.contains(&vertex)
            });
            if split {
                paired = false;
                self.subdivide(index)?;
            }
        }
        Ok(paired)
    }
}
