#[cfg(test)]
mod test;

use crate::geometry::ntree::{
    Orthotree,
    balance::{Balance, Balancing},
    node::cell::Cell,
    pair::Pairing,
};

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
    T: Cell,
    U: Copy + Into<usize>,
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
        (0..D).filter(|&other| other != axis).any(|other| {
            (0..2).any(|beyond| {
                let last = D - axis - other;
                let toward = ((1 - side) << axis) | ((1 - beyond) << other);
                let along = [toward, toward | (1 << last)];
                let adjacent = ((1 - side) << axis) | (beyond << other);
                [adjacent, adjacent | (1 << last)].into_iter().any(|child| {
                    self[children[child]].facets()[2 * other + beyond].is_some_and(|edge| {
                        self.deep_toward(edge, &along, depth - 1) || {
                            let up = (child >> last) & 1;
                            self[edge].facets()[2 * last + up].is_some_and(|vertex| {
                                self.deep_toward(vertex, &[toward | ((1 - up) << last)], depth - 1)
                            })
                        }
                    })
                })
            })
        })
    }
}

impl<T, U, V> Balance for Orthotree<D, L, M, N, T, U, V>
where
    T: Cell,
    U: Copy + From<usize> + Into<usize>,
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
                if !self[index.into()].is_unit() && self[index.into()].is_leaf() {
                    'faces: for (face, face_cell) in self[index.into()].facets().iter().enumerate()
                    {
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
