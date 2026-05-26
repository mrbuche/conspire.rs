use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, sentinel::Sentinel, split::Split},
};
use std::{array::from_fn, collections::HashSet, ops::AddAssign};

#[derive(Clone, Copy)]
pub enum Pairing {
    Regular,
    None,
}

impl<T, U> Orthotree<3, 6, 8, T, U>
where
    T: AddAssign + Copy + PartialEq + Split,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel + std::default::Default,
{
    pub fn subdivide(&mut self, index: U, pairing: Pairing) -> Result<(), OrthotreeError> {
        match pairing {
            Pairing::None => self.subdivide_node(index.into()),
            Pairing::Regular => {
                let length = self[index].length;
                let start: usize = index.into();
                let mut seen: HashSet<usize> = HashSet::from([start]);
                let mut siblings: Vec<usize> = vec![start];
                let mut i = 0;
                while i < siblings.len() {
                    let current = siblings[i];
                    for &neighbor in &self.nodes[current].facets {
                        if neighbor != U::MAX {
                            let n: usize = neighbor.into();
                            if self.nodes[n].length == length
                                && self.nodes[n].is_leaf()
                                && !seen.contains(&n)
                            {
                                seen.insert(n);
                                siblings.push(n);
                            }
                        }
                    }
                    i += 1;
                }
                for sibling in siblings {
                    self.subdivide_node(sibling)?;
                }
                Ok(())
            }
        }
    }
    fn subdivide_node(&mut self, index: usize) -> Result<(), OrthotreeError> {
        let first = self.nodes.len();
        let indices = from_fn(|n| (first + n).into());
        let parent_facets = self.nodes[index].facets;
        let mut new_nodes = self.nodes[index].subdivide(indices)?;
        for (f, &n_parent) in parent_facets.iter().enumerate() {
            if n_parent == U::MAX {
                continue;
            }
            let n_idx: usize = n_parent.into();
            if !self.nodes[n_idx].is_tree() {
                continue;
            }
            let axis = f / 2;
            let sign = f % 2;
            let opposite_f = axis * 2 + (1 - sign);
            for (ci, node) in new_nodes.iter_mut().enumerate() {
                if (ci >> axis) & 1 == sign {
                    let n_child = self.nodes[n_idx].orthants()[ci ^ (1 << axis)];
                    node.facets[f] = n_child;
                    self.nodes[n_child.into()].facets[opposite_f] = U::from(first + ci);
                }
            }
        }
        self.nodes.extend(new_nodes);
        self.nodes[index].kind = Kind::Tree(indices);
        Ok(())
    }
}

// impl<const D: usize, const M: usize, const N: usize, T, U> Orthotree<D, M, N, T, U>
// where
//     T: AddAssign + Copy + PartialEq + Split,
//     U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
// {
//     pub fn subdivide(&mut self, index: U, pairing: Pairing) -> Result<(), OrthotreeError> {
//         match pairing {
//             Pairing::None => self.subdivide_node(index.into()),
//             Pairing::Regular => {
//                 let length = self[index].length;
//                 let start: usize = index.into();
//                 let mut seen: HashSet<usize> = HashSet::from([start]);
//                 let mut siblings: Vec<usize> = vec![start];
//                 let mut i = 0;
//                 while i < siblings.len() {
//                     let current = siblings[i];
//                     for &neighbor in &self.nodes[current].facets {
//                         if neighbor != U::MAX {
//                             let n: usize = neighbor.into();
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
//                 for sibling in siblings {
//                     self.subdivide_node(sibling)?;
//                 }
//                 Ok(())
//             }
//         }
//     }
//     fn subdivide_node(&mut self, index: usize) -> Result<(), OrthotreeError> {
//         let first = self.nodes.len();
//         let indices: [U; N] = from_fn(|n| (first + n).into());
//         let parent_facets = self.nodes[index].facets;
//         let mut new_nodes = self.nodes[index].subdivide(indices)?;
//         for (f, &n_parent) in parent_facets.iter().enumerate() {
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
//             for (ci, node) in new_nodes.iter_mut().enumerate() {
//                 if (ci >> axis) & 1 == sign {
//                     let n_child = self.nodes[n_idx].orthants()[ci ^ (1 << axis)];
//                     node.facets[f] = n_child;
//                     self.nodes[n_child.into()].facets[opposite_f] = U::from(first + ci);
//                 }
//             }
//         }
//         self.nodes.extend(new_nodes);
//         self.nodes[index].kind = Kind::Tree(indices);
//         Ok(())
//     }
// }
