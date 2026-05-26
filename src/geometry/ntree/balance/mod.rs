use crate::geometry::ntree::{
    Orthotree,
    node::{sentinel::Sentinel, split::Split},
    subdivide::Pairing,
};
use std::ops::AddAssign;

pub enum Balancing {
    Faces,
    All,
}

impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
where
    T: AddAssign + Copy + Split + std::fmt::Display + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn balance(&mut self, balancing: Balancing) {
        let mut queue: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| self.nodes[i].is_leaf())
            .collect();
        while let Some(index) = queue.pop() {
            if !self.nodes[index].is_leaf() || self.nodes[index].length.into() > 1 {
                continue;
            }
            if self.violates_balance(index, &balancing) {
                let first = self.nodes.len();
                self.subdivide(U::from(index), Pairing::None).ok();
                queue.extend(first..self.nodes.len());
            }
        }
    }
    fn violates_balance(&self, index: usize, balancing: &Balancing) -> bool {
        for f in 0..M {
            let n = self.nodes[index].facets[f];
            if n == U::MAX {
                continue;
            }
            let n_idx = n.into();
            if !self.nodes[n_idx].is_tree() {
                continue;
            }
            let axis = f / 2;
            let mirror_sign = 1 - f % 2;
            for ci in 0..N {
                if (ci >> axis) & 1 != mirror_sign {
                    continue;
                }
                let child = self.nodes[n_idx].orthants()[ci];
                if self.nodes[child.into()].is_tree() {
                    return true;
                }
                if matches!(balancing, Balancing::All)
                    && self.has_diagonal_tree(child.into(), ci, 1 << axis)
                {
                    return true;
                }
            }
        }
        false
    }
    fn has_diagonal_tree(&self, current: usize, ci: usize, visited: usize) -> bool {
        for b in 0..D {
            if (visited >> b) & 1 == 1 {
                continue;
            }
            let facet = b * 2 + ((ci >> b) & 1);
            let next = self.nodes[current].facets[facet];
            if next == U::MAX {
                continue;
            }
            let next_idx = next.into();
            if self.nodes[next_idx].is_tree() {
                return true;
            }
            if self.has_diagonal_tree(next_idx, ci, visited | (1 << b)) {
                return true;
            }
        }
        false
    }
}
