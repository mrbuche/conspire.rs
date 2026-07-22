mod ilp;
#[cfg(test)]
mod test;

use crate::geometry::ntree::{Orthotree, node::split::Split};
use ilp::Instance;
use std::{collections::HashSet, ops::Add};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    pub(super) fn pair_generalized(&mut self) -> Result<bool, &'static str> {
        let lengths: HashSet<usize> = (0..self.len())
            .filter(|&index| self[index.into()].is_leaf())
            .map(|index| self[index.into()].length.into())
            .collect();
        let coarse = match lengths.len() {
            0 | 1 => return Ok(true),
            2 => *lengths.iter().max().unwrap(),
            _ => return Err("generalized pairing supports at most two refinement levels"),
        };
        let fine = *lengths.iter().min().unwrap();
        if coarse != 2 * fine {
            return Err("generalized pairing supports at most two refinement levels");
        }
        let mut coarse_nodes = Vec::new();
        for index in 0..self.len() {
            let idx: U = index.into();
            if self[idx].length.into() != coarse {
                continue;
            }
            let required = self[idx].is_tree();
            if required {
                let orthants = *self[idx].orthants().unwrap();
                for child in orthants {
                    if self[child].length.into() != fine || !self[child].is_leaf() {
                        return Err("generalized pairing supports at most two refinement levels");
                    }
                }
            }
            let corner: [i32; D] = self[idx]
                .corner
                .map(|coordinate| (coordinate.into() / coarse) as i32);
            coarse_nodes.push((idx, corner, required));
        }
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
