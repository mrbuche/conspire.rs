use crate::geometry::ntree::{
    Orthotree,
    node::{sentinel::Sentinel, split::Split},
    subdivide::Pairing,
    error::OrthotreeError
};
use std::{array::from_fn, ops::Add};

const NUM_EDGES: usize = 8;
const NUM_FACES: usize = 6;
const NUM_OCTANTS: usize = 8;

pub enum Balancing {
    Faces,
    All,
}

impl<T, U> Orthotree<3, 6, 8, T, U>
where
    T: Add<Output = T> + Copy + PartialEq + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn balance_and_pair(&mut self, strong: bool) -> Result<(), OrthotreeError> {
        let mut balanced = false;
        let mut paired = false;
        while !balanced || !paired {
            balanced = self.balance(strong);
            paired = self.pair()?;
        }
        Ok(())
    }
    pub fn balance(&mut self, strong: bool) -> bool {
        let mut balanced;
        let mut balanced_already = true;
        let mut edges = [false; NUM_EDGES];
        let mut index;
        let mut subdivide;
        let mut vertices = [false; 2];
        #[allow(unused_variables)]
        for iteration in 1.. {
            balanced = true;
            index = 0;
            subdivide = false;
            // #[cfg(feature = "profile")]
            // let time = Instant::now();
            while index < self.nodes.len() {
                if !self[index.into()].is_voxel() && self[index.into()].is_leaf() {
                    'faces: for (face, face_cell) in
                        self[index.into()].get_faces().iter().enumerate()
                    {
                        if let Some(neighbor) = face_cell
                            && let Some(kids) = self[*neighbor].orthants()
                        {
                            if strong {
                                edges = from_fn(|_| false);
                                vertices = from_fn(|_| false);
                            }
                            if match face {
                                0 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[2] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[2] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[5] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[5] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[3]
                                            {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[3] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[3] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[4] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[4] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[3]
                                            {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    self[kids[1]].is_tree()
                                        || self[kids[3]].is_tree()
                                        || self[kids[5]].is_tree()
                                        || self[kids[7]].is_tree()
                                        || edges.into_iter().any(|edge| edge)
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                1 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[3] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[3] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[5] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[2]
                                            {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[5] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[2] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[2] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[4] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[2]
                                            {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[4] {
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    self[kids[0]].is_tree()
                                        || self[kids[2]].is_tree()
                                        || self[kids[4]].is_tree()
                                        || self[kids[6]].is_tree()
                                        || edges.into_iter().any(|edge| edge)
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                2 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[5] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[0]
                                            {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[5] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[4] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[0]
                                            {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[4] {
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    self[kids[2]].is_tree()
                                        || self[kids[3]].is_tree()
                                        || self[kids[6]].is_tree()
                                        || self[kids[7]].is_tree()
                                        || edges.into_iter().any(|edge: bool| edge)
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                3 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[0] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[0] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[5] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[5] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[1]
                                            {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[1] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[1] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[4] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[4] {
                                            if let Some(vertex_cell) =
                                                self[edge_cell].get_faces()[1]
                                            {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    self[kids[0]].is_tree()
                                        || self[kids[1]].is_tree()
                                        || self[kids[4]].is_tree()
                                        || self[kids[5]].is_tree()
                                        || edges.into_iter().any(|edge| edge)
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                4 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[3] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].get_faces()[3] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].get_faces()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].get_faces()[2] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].get_faces()[2] {
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    edges.into_iter().any(|edge| edge)
                                        || self[kids[4]].is_tree()
                                        || self[kids[5]].is_tree()
                                        || self[kids[6]].is_tree()
                                        || self[kids[7]].is_tree()
                                }
                                5 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[3] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].get_faces()[3] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].get_faces()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].get_faces()[2] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].get_faces()[2] {
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    edges.into_iter().any(|edge| edge)
                                        || self[kids[0]].is_tree()
                                        || self[kids[1]].is_tree()
                                        || self[kids[2]].is_tree()
                                        || self[kids[3]].is_tree()
                                }
                                _ => panic!(),
                            } {
                                subdivide = true;
                                break 'faces;
                            }
                        }
                    }
                    if subdivide {
                        self.subdivide(index.into(), Pairing::None).unwrap();
                        balanced = false;
                        balanced_already = false;
                        subdivide = false;
                    }
                }
                index += 1;
            }
            // #[cfg(feature = "profile")]
            // println!(
            //     "             \x1b[1;93mBalancing iteration {}\x1b[0m {:?} ",
            //     iteration,
            //     time.elapsed()
            // );
            if balanced {
                break;
            }
        }
        balanced_already
    }
    pub fn pair(&mut self) -> Result<bool, OrthotreeError> {
        let mut index = 0;
        let mut paired = true;
        while index < self.nodes.len() {
            if let Some(nodes) = self[index.into()].orthants() {
                let any_tree = nodes.iter().any(|&subcell| self[subcell].is_tree());
                let all_tree = nodes.iter().all(|&subcell| self[subcell].is_tree());
                if any_tree && !all_tree {
                    let leaves: Vec<_> = nodes
                        .iter()
                        .copied()
                        .filter(|&subcell| self[subcell].is_leaf())
                        .collect();
                    for node in leaves {
                        paired = false;
                        self.subdivide(node, Pairing::None)?;
                    }
                }
            }
            index += 1;
        }
        Ok(paired)
    }

    // fn violates_balance(&self, index: usize, balancing: &Balancing) -> bool {
    //     for f in 0..6_usize {
    //         let n = self.nodes[index].facets[f];
    //         if n == U::MAX {
    //             continue;
    //         }
    //         let n_idx: usize = n.into();
    //         if !self.nodes[n_idx].is_tree() {
    //             continue;
    //         }
    //         let axis = f / 2;
    //         let mirror_sign = 1 - f % 2;
    //         for ci in 0..8_usize {
    //             if (ci >> axis) & 1 != mirror_sign {
    //                 continue;
    //             }
    //             let child = self.nodes[n_idx].orthants()[ci];
    //             if self.nodes[child.into()].is_tree() {
    //                 return true;
    //             }
    //             if matches!(balancing, Balancing::All)
    //                 && self.has_diagonal_tree(child.into(), ci, 1 << axis)
    //             {
    //                 return true;
    //             }
    //         }
    //     }
    //     false
    // }

    // fn has_diagonal_tree(&self, current: usize, ci: usize, visited: usize) -> bool {
    //     for b in 0..3_usize {
    //         if (visited >> b) & 1 == 1 {
    //             continue;
    //         }
    //         let facet_idx = b * 2 + ((ci >> b) & 1);
    //         let next = self.nodes[current].facets[facet_idx];
    //         if next == U::MAX {
    //             continue;
    //         }
    //         let next_idx: usize = next.into();
    //         if self.nodes[next_idx].is_tree() {
    //             return true;
    //         }
    //         if self.has_diagonal_tree(next_idx, ci, visited | (1 << b)) {
    //             return true;
    //         }
    //     }
    //     false
    // }
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
