use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{Kind, sentinel::Sentinel, split::Split},
};
use std::{array::from_fn, ops::Add};

#[derive(Clone, Copy)]
pub enum Pairing {
    Generalized,
    Regular,
    None,
}

const fn mirror_facet(facet: usize) -> usize {
    facet ^ 1
}

const fn insert_bit(x: usize, axis: usize, bit: usize) -> usize {
    let low_mask = (1usize << axis) - 1;
    let low = x & low_mask;
    let high = x >> axis;
    low | (bit << axis) | (high << (axis + 1))
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
where
    T: Add<Output = T> + Copy + PartialEq + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    fn nodes_on_face(facet: usize) -> [usize; L] {
        from_fn(|k| insert_bit(k, facet / 2, facet % 2))
    }
    fn nodes_on_other_face(face: usize) -> [usize; L] {
        Self::nodes_on_face(mirror_facet(face))
    }
    pub fn subdivide(&mut self, index: U) -> Result<(), OrthotreeError> {
        let indices = from_fn(|n| (self.nodes.len() + n).into());
        let mut orthants = self[index].subdivide(indices)?;
        for (facet, node_facet) in self[index].facets.into_iter().enumerate() {
            if let Some(facet_node) = node_facet
                && let Some(neighbors) = self[facet_node].orthants().copied()
            {
                for (node, neighbor) in Self::nodes_on_face(facet)
                    .into_iter()
                    .zip(Self::nodes_on_other_face(facet))
                {
                    if orthants[node].facets[facet].is_none() {
                        orthants[node].facets[facet] = Some(neighbors[neighbor])
                    } else {
                        panic!("temporary to assess need for Option<>")
                    }
                    if self[neighbors[neighbor]].facets[mirror_facet(facet)].is_none() {
                        self[neighbors[neighbor]].facets[mirror_facet(facet)] = Some(indices[node])
                    } else {
                        panic!("temporary to assess need for Option<>")
                    }
                }
            }
        }
        self.nodes.extend(orthants);
        self[index].kind = Kind::Tree(indices);
        Ok(())
    }
}

// use crate::geometry::ntree::{
//     Orthotree,
//     error::OrthotreeError,
//     node::{Kind, sentinel::Sentinel, split::Split},
// };
// use std::{array::from_fn, collections::HashSet, ops::Add};

// const NUM_SUBCELLS_FACE: usize = 4;
// type SubcellsOnFace = [usize; NUM_SUBCELLS_FACE];

// const SUBCELLS_ON_OWN_FACE_0: SubcellsOnFace = [0, 2, 4, 6]; // -x
// const SUBCELLS_ON_OWN_FACE_1: SubcellsOnFace = [1, 3, 5, 7]; // +x
// const SUBCELLS_ON_OWN_FACE_2: SubcellsOnFace = [0, 1, 4, 5]; // -y
// const SUBCELLS_ON_OWN_FACE_3: SubcellsOnFace = [2, 3, 6, 7]; // +y
// const SUBCELLS_ON_OWN_FACE_4: SubcellsOnFace = [0, 1, 2, 3]; // -z
// const SUBCELLS_ON_OWN_FACE_5: SubcellsOnFace = [4, 5, 6, 7]; // +z

// const fn subcells_on_own_face(face: usize) -> SubcellsOnFace {
//     match face {
//         0 => SUBCELLS_ON_OWN_FACE_0,
//         1 => SUBCELLS_ON_OWN_FACE_1,
//         2 => SUBCELLS_ON_OWN_FACE_2,
//         3 => SUBCELLS_ON_OWN_FACE_3,
//         4 => SUBCELLS_ON_OWN_FACE_4,
//         5 => SUBCELLS_ON_OWN_FACE_5,
//         _ => panic!(),
//     }
// }

// const fn subcells_on_neighbor_face(face: usize) -> SubcellsOnFace {
//     match face {
//         0 => SUBCELLS_ON_OWN_FACE_1,
//         1 => SUBCELLS_ON_OWN_FACE_0,
//         2 => SUBCELLS_ON_OWN_FACE_3,
//         3 => SUBCELLS_ON_OWN_FACE_2,
//         4 => SUBCELLS_ON_OWN_FACE_5,
//         5 => SUBCELLS_ON_OWN_FACE_4,
//         _ => panic!(),
//     }
// }

// const fn mirror_face(face: usize) -> usize {
//     match face {
//         0 => 1,
//         1 => 0,
//         2 => 3,
//         3 => 2,
//         4 => 5,
//         5 => 4,
//         _ => panic!(),
//     }
// }

// #[derive(Clone, Copy)]
// pub enum Pairing {
//     Regular,
//     None,
// }

// impl<T, U> Orthotree<3, 6, 8, T, U>
// where
//     T: Add<Output = T> + Copy + PartialEq + Split + Into<usize>,
//     U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
// {
//     pub fn subdivide(&mut self, index: U) -> Result<(), OrthotreeError> {
//         let new_indices = from_fn(|n| (self.nodes.len() + n).into());
//         let mut new_cells = self[index].subdivide(new_indices)?;
//         self[index]
//             .facets
//             .clone()
//             .iter()
//             .enumerate()
//             .for_each(|(face, &face_cell)| {
//                 if let Some(neighbor) = face_cell
//                     && let Some(kids) = self[neighbor].orthants().copied()
//                 {
//                     subcells_on_own_face(face)
//                         .iter()
//                         .zip(subcells_on_neighbor_face(face).iter())
//                         .for_each(|(&subcell, &neighbor_subcell)| {
//                             new_cells[subcell].facets[face] = Some(kids[neighbor_subcell]);
//                             self[kids[neighbor_subcell]].facets[mirror_face(face)] =
//                                 Some(new_indices[subcell]);
//                         });
//                 }
//             });
//         self.nodes.extend(new_cells);
//         self[index].kind = Kind::Tree(new_indices);
//         Ok(())
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
// }
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
// }

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
