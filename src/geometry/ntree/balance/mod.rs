// use crate::geometry::ntree::{
//     Orthotree,
//     node::{sentinel::Sentinel, split::Split},
//     subdivide::Pairing,
// };
// use std::ops::AddAssign;

pub enum Balancing {
    Faces,
    All,
}

// impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
// where
//     T: AddAssign + Copy + Default + PartialEq + Split,
//     U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
// {
//     pub fn balance(&mut self, balancing: Balancing) {
//         let mut queue: Vec<usize> = (0..self.nodes.len())
//             .filter(|&i| self.nodes[i].is_leaf())
//             .collect();
//         while let Some(index) = queue.pop() {
//             if !self.nodes[index].is_leaf() {
//                 continue;
//             }
//             if self.violates_balance(index, &balancing) {
//                 let first = self.nodes.len();
//                 self.subdivide(index, Pairing::Regular).ok();
//                 queue.extend(first..self.nodes.len());
//             }
//         }
//     }
//     fn violates_balance(&self, index: usize, balancing: &Balancing) -> bool {
//         for f in 0..M {
//             let n = self.nodes[index].facets[f];
//             if n == U::MAX {
//                 continue;
//             }
//             if !self.nodes[n.into()].is_tree() {
//                 continue;
//             }
//             let axis = f / 2;
//             let mirror_sign = 1 - f % 2;
//             let facing: Vec<U> = (0..N)
//                 .filter(|&i| (i >> axis) & 1 == mirror_sign)
//                 .map(|i| self.nodes[n.into()].orthants()[i])
//                 .collect();
//             if facing.iter().any(|&c| self.nodes[c.into()].is_tree()) {
//                 return true;
//             }
//             if matches!(balancing, Balancing::All) {
//                 for &c in &facing {
//                     for g in (0..M).filter(|&g| g / 2 != axis) {
//                         let e = self.nodes[c.into()].facets[g];
//                         if e != U::MAX && self.nodes[e.into()].is_tree() {
//                             return true;
//                         }
//                     }
//                 }
//             }
//         }
//         false
//     }
// }
