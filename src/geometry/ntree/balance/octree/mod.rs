#[cfg(test)]
mod test;

use crate::geometry::ntree::{
    Orthotree,
    balance::{Balance, Balancing},
    node::split::Split,
    pair::Pairing,
};
use std::{array::from_fn, ops::Add};

const NUM_EDGES: usize = 8;

const D: usize = 3;
const L: usize = 4;
const M: usize = 6;
const N: usize = 8;

const FACE_ORTHANTS: [[usize; 4]; M] = [
    [1, 3, 5, 7],
    [0, 2, 4, 6],
    [2, 3, 6, 7],
    [0, 1, 4, 5],
    [4, 5, 6, 7],
    [0, 1, 2, 3],
];

impl<T, U, V> Orthotree<D, L, M, N, T, U, V>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    fn deep(&self, cell: U, face: usize, depth: usize) -> bool {
        match self[cell].orthants() {
            None => false,
            Some(orthants) => {
                depth == 0
                    || FACE_ORTHANTS[face]
                        .iter()
                        .any(|&orthant| self.deep(orthants[orthant], face, depth - 1))
            }
        }
    }
}

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
        let mut edges = [false; NUM_EDGES];
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
                            if let Balancing::Weak(depth) = balancing {
                                if self.deep(*neighbor, face, depth) {
                                    subdivide = true;
                                    break 'faces;
                                }
                                continue;
                            }
                            if strong {
                                edges = from_fn(|_| false);
                                vertices = from_fn(|_| false);
                            }
                            if match face {
                                0 => {
                                    if strong {
                                        if let Some(edge_cell) = self[kids[1]].facets()[2] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].facets()[2] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].facets()[5] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[5] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[3] {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].facets()[3] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[3] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].facets()[4] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].facets()[4] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[3] {
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
                                        if let Some(edge_cell) = self[kids[2]].facets()[3] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[3] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[5] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[2] {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[5] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].facets()[2] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[2] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].facets()[4] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[2] {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].facets()[4] {
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
                                        if let Some(edge_cell) = self[kids[3]].facets()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[5] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[0] {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[5] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].facets()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].facets()[4] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[0] {
                                                vertices[1] = self[vertex_cell].is_tree()
                                            }
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].facets()[4] {
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
                                        if let Some(edge_cell) = self[kids[0]].facets()[0] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[0] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[5] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].facets()[5] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[1] {
                                                vertices[0] = self[vertex_cell].is_tree()
                                            }
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].facets()[1] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].facets()[1] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].facets()[4] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].facets()[4] {
                                            if let Some(vertex_cell) = self[edge_cell].facets()[1] {
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
                                        if let Some(edge_cell) = self[kids[5]].facets()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[3] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[7]].facets()[3] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[6]].facets()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[4]].facets()[2] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[5]].facets()[2] {
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
                                        if let Some(edge_cell) = self[kids[1]].facets()[1] {
                                            edges[0] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].facets()[1] {
                                            edges[1] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].facets()[3] {
                                            edges[2] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[3]].facets()[3] {
                                            edges[3] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].facets()[0] {
                                            edges[4] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[2]].facets()[0] {
                                            edges[5] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[0]].facets()[2] {
                                            edges[6] = self[edge_cell].is_tree()
                                        }
                                        if let Some(edge_cell) = self[kids[1]].facets()[2] {
                                            edges[7] = self[edge_cell].is_tree()
                                        }
                                    }
                                    edges.into_iter().any(|edge| edge)
                                        || self[kids[0]].is_tree()
                                        || self[kids[1]].is_tree()
                                        || self[kids[2]].is_tree()
                                        || self[kids[3]].is_tree()
                                }
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
