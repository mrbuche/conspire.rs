pub mod octree;
pub mod quadtree;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::Mesh,
        ntree::{
            Orthotree,
            balance::Balancing,
            node::{Kind, split::Split},
            pair::Pairing,
        },
    },
    math::{Scalar, TensorVec},
};
use std::{array::from_fn, collections::HashMap, ops::Add};

type NodeMap<const D: usize> = HashMap<[usize; D], usize>;

fn get_or_add<const D: usize>(
    coordinate: Coordinate<D>,
    coordinates: &mut Coordinates<D>,
    nodes_map: &mut NodeMap<D>,
    node_index: &mut usize,
) -> usize {
    let key = from_fn(|i| (2.0 * coordinate[i]) as usize);
    if let Some(&node) = nodes_map.get(&key) {
        node
    } else {
        let node = *node_index;
        coordinates.push(coordinate);
        nodes_map.insert(key, node);
        *node_index += 1;
        node
    }
}

pub trait Dualization<const D: usize> {
    fn dualize(&mut self) -> Mesh<D>;
}

pub trait Star<const D: usize, const N: usize> {
    fn star(&self, center_nodes: &[usize], connectivity: &mut Vec<[usize; N]>);
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U> Star<D, N>
    for Orthotree<D, L, M, N, T, U>
where
    T: Add<Output = T> + Copy + PartialOrd + Split + Into<usize>,
    U: Copy + Into<usize>,
{
    fn star(&self, center_nodes: &[usize], connectivity: &mut Vec<[usize; N]>) {
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let root = &self.nodes[0];
        let lo = root.corner;
        let hi: [T; D] = from_fn(|a| root.corner[a] + root.length);
        for node in self.iter().filter(|node| node.is_leaf()) {
            let vertex: [T; D] = from_fn(|a| node.corner[a] + node.length);
            if (0..D).all(|a| lo[a] < vertex[a] && vertex[a] < hi[a]) {
                let cells: [usize; N] = from_fn(|d| incident_leaf(self, &vertex, d));
                let mut distinct = cells.to_vec();
                distinct.sort_unstable();
                distinct.dedup();
                if distinct.len() != N {
                    continue;
                }
                let lengths: [usize; N] = from_fn(|o| self.nodes[cells[o]].length.into());
                let shortest = *lengths.iter().min().unwrap();
                let longest = *lengths.iter().max().unwrap();
                let coordinate: [usize; D] = from_fn(|a| vertex[a].into());
                if longest == shortest || (0..D).all(|a| coordinate[a].is_multiple_of(2 * longest))
                {
                    connectivity.push(from_fn(|i| {
                        let bits = i & face_mask;
                        center_nodes[cells[(i & !face_mask) | (bits ^ (bits >> 1))]]
                    }));
                }
            }
        }
    }
}

pub(crate) fn incident_leaf<const D: usize, const L: usize, const M: usize, const N: usize, T, U>(
    tree: &Orthotree<D, L, M, N, T, U>,
    vertex: &[T; D],
    direction: usize,
) -> usize
where
    T: Add<Output = T> + Copy + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half = tree.nodes[index].length.split();
                let child = (0..D).fold(0, |acc, a| {
                    let mid = corner[a] + half;
                    let bit = if vertex[a] > mid {
                        1
                    } else if vertex[a] < mid {
                        0
                    } else {
                        (direction >> a) & 1
                    };
                    acc | (bit << a)
                });
                index = orthants[child].into();
            }
        }
    }
}

pub trait Initialize<const D: usize, const N: usize> {
    fn initialize(&self) -> (Vec<usize>, Coordinates<D>, usize, Vec<[usize; N]>);
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U> Initialize<D, N>
    for Orthotree<D, L, M, N, T, U>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    fn initialize(&self) -> (Vec<usize>, Coordinates<D>, usize, Vec<[usize; N]>) {
        assert!(matches!(
            self.balanced,
            Balancing::Strong | Balancing::Weak(1)
        ));
        assert!(!matches!(self.paired, Pairing::None));
        let num = self.len();
        let mut center_nodes = vec![0; num];
        let mut coordinates = Coordinates::with_capacity(num);
        let mut node_index = 0;
        self.iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .for_each(|(index, leaf)| {
                center_nodes[index] = node_index;
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
}
