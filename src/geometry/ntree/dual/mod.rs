pub(super) mod octree;
pub(super) mod quadtree;

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

pub(super) trait Star<const D: usize, const N: usize> {
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
                if longest == shortest || self.cluster_corner(&coordinate, longest) {
                    connectivity.push(from_fn(|i| {
                        let bits = i & face_mask;
                        center_nodes[cells[(i & !face_mask) | (bits ^ (bits >> 1))]]
                    }));
                }
            }
        }
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
{
    /// Index of the leaf with exactly this `corner` and `length`, if one exists inside the
    /// root. Absence means either off-domain or a cell of some other size covering the spot;
    /// callers that care which must test the bounds themselves.
    pub(super) fn cell_at(&self, corner: &[i64; D], length: i64) -> Option<usize>
    where
        T: Copy + Into<usize> + Split,
        U: Copy + Into<usize>,
    {
        if self.off_domain(corner, length) {
            return None;
        }
        let point = from_fn(|axis| corner[axis] as usize);
        let index = leaf_containing(self, &point);
        let node = &self.nodes[index];
        (length as usize == node.length.into()
            && (0..D).all(|axis| point[axis] == node.corner[axis].into()))
        .then_some(index)
    }
    /// Whether the cell of this `corner` and `length` reaches outside the root. This is the only
    /// reason a template may treat a missing cell as truncation: missing for any other reason
    /// means the transition there belongs to something else.
    pub(super) fn off_domain(&self, corner: &[i64; D], length: i64) -> bool
    where
        T: Copy + Into<usize>,
        U: Copy + Into<usize>,
    {
        let root = &self.nodes[0];
        let extent = Into::<usize>::into(root.length) as i64;
        (0..D).any(|axis| {
            let low = Into::<usize>::into(root.corner[axis]) as i64;
            corner[axis] < low || corner[axis] + length > low + extent
        })
    }
    /// Whether the two cells of `length` stacked along `axis` from `corner` belong to the same
    /// paired cluster. Under `Pairing::Regular` this is just "the two are siblings", but stated
    /// in terms of the pairing it holds for `Pairing::Generalized` too.
    pub(super) fn shares_cluster(&self, corner: &[i64; D], length: i64, axis: usize) -> bool {
        (0..1usize << D).any(|bits| {
            let mut center = [0; D];
            for (index, coordinate) in center.iter_mut().enumerate() {
                let shifted = if index == axis {
                    corner[index] + length
                } else {
                    corner[index] + ((bits >> index) & 1) as i64 * length
                };
                match usize::try_from(shifted) {
                    Ok(value) => *coordinate = value,
                    Err(_) => return false,
                }
            }
            self.pairing_vertices.contains(&(center, length as usize))
        })
    }
    /// Whether `vertex` is a corner of a paired cluster of cells of `length`.
    pub(super) fn cluster_corner(&self, vertex: &[usize; D], length: usize) -> bool {
        (0..1usize << D).any(|bits| {
            let mut center = [0; D];
            for (axis, coordinate) in center.iter_mut().enumerate() {
                if (bits >> axis) & 1 == 1 {
                    *coordinate = vertex[axis] + length;
                } else if let Some(shifted) = vertex[axis].checked_sub(length) {
                    *coordinate = shifted;
                } else {
                    return false;
                }
            }
            self.pairing_vertices.contains(&(center, length))
        })
    }
}

/// Index of the leaf containing `point`, with ties resolved toward increasing coordinates.
pub(super) fn leaf_containing<
    const D: usize,
    const L: usize,
    const M: usize,
    const N: usize,
    T,
    U,
>(
    tree: &Orthotree<D, L, M, N, T, U>,
    point: &[usize; D],
) -> usize
where
    T: Copy + Into<usize> + Split,
    U: Copy + Into<usize>,
{
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half: usize = tree.nodes[index].length.split().into();
                let child = (0..D).fold(0, |acc, a| {
                    let mid: usize = corner[a].into() + half;
                    acc | (usize::from(point[a] >= mid) << a)
                });
                index = orthants[child].into();
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

pub(super) trait Initialize<const D: usize, const N: usize> {
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
            Balancing::Strong(1) | Balancing::Weak(1)
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
