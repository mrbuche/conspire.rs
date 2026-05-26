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

impl<T, U> Orthotree<3, 6, 8, T, U>
where
    T: AddAssign + Copy + PartialEq + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn balance(&mut self, balancing: Balancing) {
        let mut any_subdivision = true;
        while any_subdivision {
            any_subdivision = false;
            let mut index = 0;
            while index < self.nodes.len() {
                if self.nodes[index].is_leaf()
                    && self.nodes[index].length.into() > 1
                    && self.violates_balance(index, &balancing)
                {
                    self.subdivide(U::from(index), Pairing::None).ok();
                    any_subdivision = true;
                }
                index += 1;
            }
        }
    }

    fn violates_balance(&self, index: usize, balancing: &Balancing) -> bool {
        for f in 0..6_usize {
            let n = self.nodes[index].facets[f];
            if n == U::MAX {
                continue;
            }
            let n_idx: usize = n.into();
            if !self.nodes[n_idx].is_tree() {
                continue;
            }
            let axis = f / 2;
            let mirror_sign = 1 - f % 2;
            for ci in 0..8_usize {
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
        for b in 0..3_usize {
            if (visited >> b) & 1 == 1 {
                continue;
            }
            let facet_idx = b * 2 + ((ci >> b) & 1);
            let next = self.nodes[current].facets[facet_idx];
            if next == U::MAX {
                continue;
            }
            let next_idx: usize = next.into();
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

// impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
// where
//     T: AddAssign + Copy + PartialEq + Split + Into<usize>,
//     U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
// {
//     pub fn balance(&mut self, balancing: Balancing) {
//         let mut any_subdivision = true;
//         while any_subdivision {
//             any_subdivision = false;
//             let mut index = 0;
//             while index < self.nodes.len() {
//                 if self.nodes[index].is_leaf()
//                     && self.nodes[index].length.into() > 1
//                     && self.violates_balance(index, &balancing)
//                 {
//                     self.subdivide(U::from(index), Pairing::None).ok();
//                     any_subdivision = true;
//                 }
//                 index += 1;
//             }
//         }
//     }
//     fn violates_balance(&self, index: usize, balancing: &Balancing) -> bool {
//         for f in 0..M {
//             let n = self.nodes[index].facets[f];
//             if n == U::MAX {
//                 continue;
//             }
//             let n_idx = n.into();
//             if !self.nodes[n_idx].is_tree() {
//                 continue;
//             }
//             let axis = f / 2;
//             let mirror_sign = 1 - f % 2;
//             for ci in 0..N {
//                 if (ci >> axis) & 1 != mirror_sign {
//                     continue;
//                 }
//                 let child = self.nodes[n_idx].orthants()[ci];
//                 if self.nodes[child.into()].is_tree() {
//                     return true;
//                 }
//                 if matches!(balancing, Balancing::All)
//                     && self.has_diagonal_tree(child.into(), ci, 1 << axis)
//                 {
//                     return true;
//                 }
//             }
//         }
//         false
//     }
//     fn has_diagonal_tree(&self, current: usize, ci: usize, visited: usize) -> bool {
//         for b in 0..D {
//             if (visited >> b) & 1 == 1 {
//                 continue;
//             }
//             let facet = b * 2 + ((ci >> b) & 1);
//             let next = self.nodes[current].facets[facet];
//             if next == U::MAX {
//                 continue;
//             }
//             let next_idx = next.into();
//             if self.nodes[next_idx].is_tree() {
//                 return true;
//             }
//             if self.has_diagonal_tree(next_idx, ci, visited | (1 << b)) {
//                 return true;
//             }
//         }
//         false
//     }
// }

