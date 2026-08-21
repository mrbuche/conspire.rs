use crate::geometry::ntree::node::slot::Slot;
#[cfg(test)]
mod test;

use crate::geometry::ntree::{
    Orthotree,
    balance::{Balance, Balancing},
    node::cell::Cell,
    pair::Pairing,
};

const D: usize = 2;
const L: usize = 2;
const M: usize = 4;
const N: usize = 4;

const FACE_ORTHANTS: [[usize; 2]; M] = [[1, 3], [0, 2], [2, 3], [0, 1]];

impl<T, U, V> Orthotree<D, L, M, N, T, U, V>
where
    T: Cell,
    U: Slot,
{
    fn deep_toward(&self, cell: U, orthants: &[usize], depth: usize) -> bool {
        match self[cell].orthants() {
            None => false,
            Some(children) => {
                depth == 0
                    || orthants
                        .iter()
                        .any(|&orthant| self.deep_toward(children[orthant], orthants, depth - 1))
            }
        }
    }
    fn deep(&self, cell: U, face: usize, depth: usize) -> bool {
        self.deep_toward(cell, &FACE_ORTHANTS[face], depth)
    }
    fn diagonally_deep(&self, children: &[U; N], face: usize, depth: usize) -> bool {
        let (axis, side) = (face / 2, face % 2);
        let other = 1 - axis;
        (0..2).any(|beyond| {
            let adjacent = ((1 - side) << axis) | (beyond << other);
            self[children[adjacent]].facets()[2 * other + beyond].is_some_and(|vertex| {
                let toward = ((1 - side) << axis) | ((1 - beyond) << other);
                self.deep_toward(vertex, &[toward], depth - 1)
            })
        })
    }
}

impl<T, U, V> Balance for Orthotree<D, L, M, N, T, U, V>
where
    T: Cell,
    U: Slot,
    V: Copy,
{
    fn balance(&mut self, balancing: Balancing) -> bool {
        self.balanced = balancing;
        let mut balanced;
        let mut balanced_already = true;
        let mut index;
        let mut subdivide;
        loop {
            balanced = true;
            index = 0;
            subdivide = false;
            while index < self.len() {
                if !self.nodes[index].is_unit() && self.nodes[index].is_leaf() {
                    'faces: for (face, face_cell) in self.nodes[index].facets().iter().enumerate() {
                        if let Some(neighbor) = face_cell
                            && let Some(children) = self[*neighbor].orthants()
                        {
                            let unbalanced = match balancing {
                                Balancing::Weak(depth) => self.deep(*neighbor, face, depth),
                                Balancing::Strong(depth) => {
                                    self.deep(*neighbor, face, depth)
                                        || self.diagonally_deep(children, face, depth)
                                }
                                Balancing::None => false,
                            };
                            if unbalanced {
                                subdivide = true;
                                break 'faces;
                            }
                        }
                    }
                    if subdivide {
                        self.subdivide(index).unwrap();
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
