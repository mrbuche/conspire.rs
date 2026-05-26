use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, Node, sentinel::Sentinel, split::Split},
};
use std::{array::from_fn, collections::HashSet, ops::AddAssign};

#[derive(Clone, Copy)]
pub enum Pairing {
    Regular,
    None,
}

impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
where
    U: From<usize>,
{
    pub fn subdivide(&mut self, index: usize, pairing: Pairing) -> Result<(), OrthotreeError> {
        let new_indices: [U; N] = from_fn(|n| (self.nodes.len() + n).into());
        todo!()
    }
}

// impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
// where
//     T: AddAssign + Copy + Default + PartialEq + Split,
//     U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
// {
//     pub fn subdivide(&mut self, index: usize, pairing: Pairing) -> Result<(), OrthotreeError> {
//         if index >= self.nodes.len() {
//             return Err(OrthotreeError::IndexOutOfBounds);
//         }
//         let Kind::Leaf = self.nodes[index].kind else {
//             return Err(OrthotreeError::IndexOutOfBounds);
//         };
//         match pairing {
//             Pairing::None => self.subdivide_node(index),
//             Pairing::Regular => {
//                 let length = self.nodes[index].length;
//                 let mut seen: HashSet<usize> = HashSet::new();
//                 let mut siblings: Vec<usize> = Vec::new();
//                 seen.insert(index);
//                 siblings.push(index);
//                 let mut i = 0;
//                 while i < siblings.len() {
//                     let current = siblings[i];
//                     for &neighbor in &self.nodes[current].facets {
//                         if neighbor != U::MAX {
//                             let n = neighbor.into();
//                             if self.nodes[n].length == length
//                                 && self.nodes[n].is_leaf()
//                                 && !seen.contains(&n)
//                             {
//                                 seen.insert(n);
//                                 siblings.push(n);
//                             }
//                         }
//                     }
//                     i += 1;
//                 }
//                 for &sibling in &siblings {
//                     self.subdivide_node(sibling)?;
//                 }
//                 Ok(())
//             }
//         }
//     }
//     fn subdivide_node(&mut self, index: usize) -> Result<(), OrthotreeError> {
//         let Kind::Leaf = self.nodes[index].kind else {
//             return Err(OrthotreeError::IndexOutOfBounds);
//         };
//         let corner = self.nodes[index].corner;
//         let length = self.nodes[index].length.split();
//         let parent_facets = self.nodes[index].facets;
//         let first = self.nodes.len();
//         for i in 0..N {
//             self.nodes.push(Node {
//                 corner: from_fn(|ax| {
//                     if (i >> ax) & 1 == 1 {
//                         let mut c = corner[ax];
//                         c += length;
//                         c
//                     } else {
//                         corner[ax]
//                     }
//                 }),
//                 length,
//                 facets: from_fn(|f| {
//                     let axis = f / 2;
//                     let sign = f % 2;
//                     if (i >> axis) & 1 == sign {
//                         parent_facets[f]
//                     } else {
//                         U::from(first + (i ^ (1 << axis)))
//                     }
//                 }),
//                 kind: Kind::Leaf,
//             });
//         }
//         self.nodes[index].kind = Kind::Tree(from_fn(|i| U::from(first + i)));
//         // If an external face neighbor is already a tree, wire children directly
//         // to that tree's specific facing children and update those children back.
//         // This mirrors what automesh does in its subdivide, keeping all face
//         // pointers accurate so balance never sees stale tree-parent references.
//         for f in 0..M {
//             let n_parent = parent_facets[f];
//             if n_parent == U::MAX {
//                 continue;
//             }
//             let n_idx: usize = n_parent.into();
//             if !self.nodes[n_idx].is_tree() {
//                 continue;
//             }
//             let axis = f / 2;
//             let sign = f % 2;
//             let opposite_f = axis * 2 + (1 - sign);
//             for ci in 0..N {
//                 if (ci >> axis) & 1 == sign {
//                     let n_child = self.nodes[n_idx].orthants()[ci ^ (1 << axis)];
//                     self.nodes[first + ci].facets[f] = n_child;
//                     self.nodes[n_child.into()].facets[opposite_f] = U::from(first + ci);
//                 }
//             }
//         }
//         Ok(())
//     }
// }
