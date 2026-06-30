use crate::geometry::ntree::{
    Orthotree,
    balance::{Balance, Balancing},
    node::split::Split,
    pair::Pairing,
};
use std::{array::from_fn, ops::Add};

const D: usize = 2;
const L: usize = 2;
const M: usize = 4;
const N: usize = 4;

impl<T, U, V> Balance for Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    fn balance(&mut self, balancing: Balancing) -> bool {
        self.balanced = balancing;
        let mut balanced;
        let mut balanced_already = true;
        let mut index;
        let mut subdivide;
        let mut vertices = [false; 2];
        let strong = matches!(balancing, Balancing::Strong);
        loop {
            balanced = true;
            index = 0;
            subdivide = false;
            while index < self.len() {
                if !self[index.into()].is_unit() && self[index.into()].is_leaf() {
                    'faces: for (face, face_cell) in self[index.into()].facets().iter().enumerate()
                    {
                        if let Some(neighbor) = face_cell
                            && let Some(kids) = self[*neighbor].orthants()
                        {
                            if strong {
                                vertices = from_fn(|_| false);
                            }
                            if match face {
                                0 => {
                                    if strong {
                                        if let Some(vertex_cell) = self[kids[1]].facets()[2] {
                                            vertices[0] = self[vertex_cell].is_tree()
                                        }
                                        if let Some(vertex_cell) = self[kids[3]].facets()[3] {
                                            vertices[1] = self[vertex_cell].is_tree()
                                        }
                                    }
                                    self[kids[1]].is_tree()
                                        || self[kids[3]].is_tree()
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                1 => {
                                    if strong {
                                        if let Some(vertex_cell) = self[kids[0]].facets()[2] {
                                            vertices[0] = self[vertex_cell].is_tree()
                                        }
                                        if let Some(vertex_cell) = self[kids[2]].facets()[3] {
                                            vertices[1] = self[vertex_cell].is_tree()
                                        }
                                    }
                                    self[kids[0]].is_tree()
                                        || self[kids[2]].is_tree()
                                        || vertices.into_iter().any(|vertex| vertex)
                                }
                                2 => self[kids[2]].is_tree() || self[kids[3]].is_tree(),
                                3 => self[kids[0]].is_tree() || self[kids[1]].is_tree(),
                                _ => unreachable!(),
                            } {
                                subdivide = true;
                                break 'faces;
                            }
                        }
                    }
                    if subdivide {
                        self.subdivide(index.into()).unwrap();
                        balanced = false;
                        balanced_already = false;
                        subdivide = false;
                    }
                }
                index += 1;
            }
            if balanced {
                break;
            }
        }
        balanced_already
    }
    fn pair_up(&mut self, pairing: Pairing) -> Result<bool, &'static str> {
        self.paired = pairing;
        self.pair(pairing)
    }
}
