use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, sentinel::Sentinel, split::Split},
};
use std::{array::from_fn, collections::HashSet, ops::AddAssign};

const NUM_SUBCELLS_FACE: usize = 4;
type SubcellsOnFace = [usize; NUM_SUBCELLS_FACE];

const SUBCELLS_ON_OWN_FACE_0: SubcellsOnFace = [0, 2, 4, 6]; // -x
const SUBCELLS_ON_OWN_FACE_1: SubcellsOnFace = [1, 3, 5, 7]; // +x
const SUBCELLS_ON_OWN_FACE_2: SubcellsOnFace = [0, 1, 4, 5]; // -y
const SUBCELLS_ON_OWN_FACE_3: SubcellsOnFace = [2, 3, 6, 7]; // +y
const SUBCELLS_ON_OWN_FACE_4: SubcellsOnFace = [0, 1, 2, 3]; // -z
const SUBCELLS_ON_OWN_FACE_5: SubcellsOnFace = [4, 5, 6, 7]; // +z

const fn subcells_on_own_face(face: usize) -> SubcellsOnFace {
    match face {
        0 => SUBCELLS_ON_OWN_FACE_0,
        1 => SUBCELLS_ON_OWN_FACE_1,
        2 => SUBCELLS_ON_OWN_FACE_2,
        3 => SUBCELLS_ON_OWN_FACE_3,
        4 => SUBCELLS_ON_OWN_FACE_4,
        5 => SUBCELLS_ON_OWN_FACE_5,
        _ => panic!(),
    }
}

const fn subcells_on_neighbor_face(face: usize) -> SubcellsOnFace {
    match face {
        0 => SUBCELLS_ON_OWN_FACE_1,
        1 => SUBCELLS_ON_OWN_FACE_0,
        2 => SUBCELLS_ON_OWN_FACE_3,
        3 => SUBCELLS_ON_OWN_FACE_2,
        4 => SUBCELLS_ON_OWN_FACE_5,
        5 => SUBCELLS_ON_OWN_FACE_4,
        _ => panic!(),
    }
}

const fn mirror_face(face: usize) -> usize {
    match face {
        0 => 1,
        1 => 0,
        2 => 3,
        3 => 2,
        4 => 5,
        5 => 4,
        _ => panic!(),
    }
}

#[derive(Clone, Copy)]
pub enum Pairing {
    Regular,
    None,
}

impl<T, U> Orthotree<3, 6, 8, T, U>
where
    T: AddAssign + Copy + PartialEq + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn subdivide(&mut self, index: U, pairing: Pairing) -> Result<(), OrthotreeError> {
        let new_indices = from_fn(|n| (self.nodes.len() + n).into());
        let mut new_cells = self[index].subdivide(new_indices)?;
        self[index]
            .facets
            .clone()
            .iter()
            .enumerate()
            .for_each(|(face, &face_cell)| {
                if let Some(neighbor) = face_cell
                    && let Some(kids) = self[neighbor].get_cells().copied()
                {
                    subcells_on_own_face(face)
                        .iter()
                        .zip(subcells_on_neighbor_face(face).iter())
                        .for_each(|(&subcell, &neighbor_subcell)| {
                            new_cells[subcell].facets[face] = Some(kids[neighbor_subcell]);
                            self[kids[neighbor_subcell]].facets[mirror_face(face)] =
                                Some(new_indices[subcell]);
                        });
                }
            });
        self.nodes.extend(new_cells);
        self[index].kind = Kind::Tree(new_indices);
        Ok(())
        // match pairing {
        //     Pairing::None => self.subdivide_node(index.into()),
        //     Pairing::Regular => {
        //         let length = self[index].length;
        //         let start: usize = index.into();
        //         let mut seen: HashSet<usize> = HashSet::from([start]);
        //         let mut siblings: Vec<usize> = vec![start];
        //         let mut i = 0;
        //         while i < siblings.len() {
        //             let current = siblings[i];
        //             for &neighbor in &self.nodes[current].facets {
        //                 if neighbor != U::MAX {
        //                     let n: usize = neighbor.into();
        //                     if self.nodes[n].length == length
        //                         && self.nodes[n].is_leaf()
        //                         && !seen.contains(&n)
        //                     {
        //                         seen.insert(n);
        //                         siblings.push(n);
        //                     }
        //                 }
        //             }
        //             i += 1;
        //         }
        //         for sibling in siblings {
        //             self.subdivide_node(sibling)?;
        //         }
        //         Ok(())
        //     }
        // }
    }
    // fn subdivide_node(&mut self, index: usize) -> Result<(), OrthotreeError> {
    //     let first = self.nodes.len();
    //     let indices = from_fn(|n| (first + n).into());
    //     let parent_facets = self.nodes[index].facets;
    //     let mut new_nodes = self.nodes[index].subdivide(indices)?;
    //     for (f, &n_parent) in parent_facets.iter().enumerate() {
    //         if n_parent == U::MAX {
    //             continue;
    //         }
    //         let n_idx: usize = n_parent.into();
    //         if !self.nodes[n_idx].is_tree() {
    //             continue;
    //         }
    //         let axis = f / 2;
    //         let sign = f % 2;
    //         let opposite_f = axis * 2 + (1 - sign);
    //         for (ci, node) in new_nodes.iter_mut().enumerate() {
    //             if (ci >> axis) & 1 == sign {
    //                 let n_child = self.nodes[n_idx].orthants()[ci ^ (1 << axis)];
    //                 node.facets[f] = n_child;
    //                 self.nodes[n_child.into()].facets[opposite_f] = U::from(first + ci);
    //             }
    //         }
    //     }
    //     self.nodes.extend(new_nodes);
    //     self.nodes[index].kind = Kind::Tree(indices);
    //     Ok(())
    // }
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
