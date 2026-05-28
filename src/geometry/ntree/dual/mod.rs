pub mod octree;
pub mod quadtree;

use crate::{
    geometry::{
        Coordinates,
        mesh::PrimitiveMesh,
        ntree::{Orthotree, balance::Balancing, pair::Pairing},
    },
    math::{Scalar, TensorVec},
};
use std::array::from_fn;

pub trait Dualization<const D: usize, const I: usize, const M: usize, const N: usize, T> {
    fn dualize(&mut self) -> PrimitiveMesh<D, I, M, N, T>;
}

pub trait Uniform<const D: usize, const N: usize, V> {
    fn initialize<const I: usize>(&self) -> (Vec<V>, Coordinates<D, I>, usize, Vec<[V; N]>);
    fn uniform_transitions(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>);
    fn uniform_transition_1(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>);
    fn uniform_transition_2(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>);
    fn uniform_transition_3(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>);
    fn uniform_transition_4(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>);
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> Uniform<D, N, V>
    for Orthotree<D, L, M, N, T, U>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy + Default + From<usize>,
{
    fn initialize<const I: usize>(&self) -> (Vec<V>, Coordinates<D, I>, usize, Vec<[V; N]>) {
        assert!(!matches!(self.balanced, Balancing::None));
        assert!(!matches!(self.paired, Pairing::None));
        let num = self.len();
        let mut center_nodes = vec![V::default(); num];
        let mut coordinates = Coordinates::with_capacity(num);
        let mut node_index = 0;
        self.iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .for_each(|(index, leaf)| {
                center_nodes[index] = V::from(node_index);
                let length: Scalar = leaf.length.into();
                let center = from_fn(|i| {
                    let c: Scalar = leaf.corner[i].into();
                    c + length * 0.5
                });
                coordinates.push(center.into());
                node_index += 1;
            });
        (
            center_nodes,
            coordinates,
            node_index,
            Vec::with_capacity(num),
        )
    }
    fn uniform_transitions(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>) {
        self.uniform_transition_1(center_nodes, connectivity);
        self.uniform_transition_2(center_nodes, connectivity);
        self.uniform_transition_3(center_nodes, connectivity);
        self.uniform_transition_4(center_nodes, connectivity);
    }
    fn uniform_transition_1(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>) {
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        connectivity.extend(
            self.iter()
                .filter_map(|node| self.all_leaves(node))
                .map(|leaves| {
                    from_fn(|i| {
                        let face = i & face_mask;
                        let vertex = (i & !face_mask) | (face ^ (face >> 1));
                        center_nodes[leaves[vertex].into()]
                    })
                }),
        )
    }
    fn uniform_transition_2(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>) {
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        self.iter().for_each(|node| {
            let leaves_and_facets = self.leaves_and_facets(node);
            for axis in 0..D {
                let low_mask = (1_usize << axis) - 1;
                let face: [Option<(U, U)>; L] = from_fn(|j| {
                    let orthant_index = ((j & !low_mask) << 1) | (j & low_mask);
                    let (leaf, facets) = leaves_and_facets[orthant_index]?;
                    let neighbor = facets[axis]?;
                    self[neighbor].is_leaf().then_some((leaf, neighbor))
                });
                if face.iter().any(|x| x.is_none()) {
                    continue;
                }
                connectivity.push(from_fn(|i| {
                    let bits = i & face_mask;
                    let vertex = (i & !face_mask) | (bits ^ (bits >> 1));
                    let side = (vertex >> axis) & 1;
                    let j = (vertex & low_mask) | ((vertex >> (axis + 1)) << axis);
                    let (leaf, neighbor) = face[j].unwrap();
                    center_nodes[if side == 0 { neighbor } else { leaf }.into()]
                }));
            }
        });
    }
    fn uniform_transition_3(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>) {
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let n_minus_1 = N - 1;
        self.iter().for_each(|node| {
            let Some(leaf_0) = self.leaves(node)[0] else {
                return;
            };
            let mut cells: [Option<U>; N] = [None; N];
            cells[0] = Some(leaf_0);
            for s in 1..N {
                let b = s.trailing_zeros() as usize;
                let prev_s = s & !(1 << b);
                if let Some(prev) = cells[prev_s]
                    && self[prev].is_leaf()
                {
                    cells[s] = self[prev].facets()[2 * b];
                }
            }
            if cells.iter().any(|c| match c {
                Some(idx) => !self[*idx].is_leaf(),
                None => true,
            }) {
                return;
            }
            connectivity.push(from_fn(|i| {
                let bits = i & face_mask;
                let vertex = (i & !face_mask) | (bits ^ (bits >> 1));
                center_nodes[cells[n_minus_1 ^ vertex].unwrap().into()]
            }));
        });
    }
    fn uniform_transition_4(&self, center_nodes: &[V], connectivity: &mut Vec<[V; N]>) {
        if D < 3 {
            return;
        }
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let n_minus_1 = N - 1;
        self.iter().for_each(|node| {
            let leaves = self.leaves(node);
            for axis_e in 0..D {
                let e_bit = 1 << axis_e;
                let non_e_mask = n_minus_1 & !e_bit;
                let Some(leaf_lo) = leaves[0] else {
                    continue;
                };
                let Some(leaf_hi) = leaves[e_bit] else {
                    continue;
                };
                let mut cells: [Option<U>; N] = [None; N];
                cells[0] = Some(leaf_lo);
                cells[e_bit] = Some(leaf_hi);
                for m in 1..N {
                    if m == e_bit {
                        continue;
                    }
                    let b = (m & non_e_mask).trailing_zeros() as usize;
                    let prev_m = m & !(1 << b);
                    if let Some(prev) = cells[prev_m]
                        && self[prev].is_leaf()
                    {
                        cells[m] = self[prev].facets()[2 * b];
                    }
                }
                if cells.iter().any(|c| match c {
                    Some(idx) => !self[*idx].is_leaf(),
                    None => true,
                }) {
                    continue;
                }
                connectivity.push(from_fn(|i| {
                    let bits = i & face_mask;
                    let vertex = (i & !face_mask) | (bits ^ (bits >> 1));
                    let m = vertex ^ non_e_mask;
                    center_nodes[cells[m].unwrap().into()]
                }));
            }
        });
    }
}
