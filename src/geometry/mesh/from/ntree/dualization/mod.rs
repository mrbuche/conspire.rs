use crate::geometry::ntree::node::slot::Slot;
pub(super) mod octree;
pub(super) mod quadtree;
#[cfg(test)]
pub(crate) use quadtree::verify_dual;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::Mesh,
        ntree::{
            Balancing, Orthotree, Pairing,
            node::{Kind, cell::Cell},
        },
    },
    math::{FxHashMap, Scalar, TensorVec},
};
use std::{array::from_fn, collections::HashMap};

type NodeMap<const D: usize> = HashMap<[usize; D], usize>;

/// `(corner, length) -> leaf index`, built once per dualization so every template's lookups are
/// O(1) instead of a fresh root-to-leaf descent each. Keyed the same way `pairing_vertices` and
/// `NodeMap` are: raw cell-grid coordinates, not tree-node indices, so a query needs no tree walk
/// at all.
pub(super) type LeafIndex<const D: usize> = FxHashMap<([usize; D], usize), usize>;

pub(super) fn build_leaf_index<
    const D: usize,
    const L: usize,
    const M: usize,
    const N: usize,
    T,
    U,
>(
    tree: &Orthotree<D, L, M, N, T, U>,
) -> LeafIndex<D>
where
    T: Cell,
{
    tree.iter()
        .enumerate()
        .filter(|(_, node)| node.is_leaf())
        .map(|(index, node)| {
            let corner: [usize; D] = from_fn(|axis| node.corner[axis].cells());
            ((corner, node.length.cells()), index)
        })
        .collect()
}

fn get_or_add<const D: usize>(
    coordinate: Coordinate<D>,
    coordinates: &mut Coordinates<D>,
    nodes_map: &mut NodeMap<D>,
    node_index: &mut usize,
) -> usize {
    let key = from_fn(|i| (2.0 * coordinate[i].value()) as usize);
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
    fn dualize(&self) -> Mesh<D>;
}

pub(super) trait Star<const D: usize, const N: usize> {
    fn star(&self, center_nodes: &[usize], connectivity: &mut Vec<[usize; N]>);
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U> Star<D, N>
    for Orthotree<D, L, M, N, T, U>
where
    T: Cell,
    U: Slot,
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
                let lengths: [usize; N] = from_fn(|o| self.nodes[cells[o]].length.cells());
                let shortest = *lengths.iter().min().unwrap();
                let longest = *lengths.iter().max().unwrap();
                let coordinate: [usize; D] = from_fn(|a| vertex[a].cells());
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
    /// callers that care which must test the bounds themselves. O(1) against a `LeafIndex`
    /// built once per dualization by `build_leaf_index`, rather than a tree descent per call.
    pub(super) fn cell_at(
        &self,
        leaf_index: &LeafIndex<D>,
        corner: &[i64; D],
        length: i64,
    ) -> Option<usize> {
        let key: ([usize; D], usize) = (from_fn(|axis| corner[axis] as usize), length as usize);
        leaf_index.get(&key).copied()
    }
    /// Whether the cell of this `corner` and `length` reaches outside the root. This is the only
    /// reason a template may treat a missing cell as truncation: missing for any other reason
    /// means the transition there belongs to something else.
    pub(super) fn off_domain(&self, corner: &[i64; D], length: i64) -> bool
    where
        T: Cell,
    {
        let root = &self.nodes[0];
        let extent = root.length.cells() as i64;
        (0..D).any(|axis| {
            let low = root.corner[axis].cells() as i64;
            corner[axis] < low || corner[axis] + length > low + extent
        })
    }
    /// Whether the two cells of `length` stacked along `axis` from `corner` belong to the same
    /// paired cluster. Under `Pairing::Regular` this is just "the two are siblings", but stated
    /// in terms of the pairing it holds for `Pairing::Generalized` too.
    pub(super) fn shares_cluster(&self, corner: &[i64; D], length: i64, axis: usize) -> bool {
        // The cluster centre is pinned along `axis`, so only the other axes vary. Enumerating all
        // `D` bits would probe each centre twice.
        (0..1usize << (D - 1)).any(|bits| {
            let mut center = [0; D];
            let mut bit = 0;
            for (index, coordinate) in center.iter_mut().enumerate() {
                let shifted = if index == axis {
                    corner[index] + length
                } else {
                    let offset = ((bits >> bit) & 1) as i64 * length;
                    bit += 1;
                    corner[index] + offset
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
pub(crate) fn leaf_containing<
    const D: usize,
    const L: usize,
    const M: usize,
    const N: usize,
    T,
    U,
    V,
>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
    point: &[usize; D],
) -> usize
where
    T: Cell,
    U: Slot,
{
    leaf_containing_from(tree, 0, point)
}

/// Same as `leaf_containing`, but descends from `start` instead of the root. Correct for any
/// `start` whose cell actually contains `point` - the caller is on the hook for that, since nothing
/// here can check it without doing the very root descent this exists to skip. Callers that already
/// know a node close to `point` (e.g. one exactly `length` away on the same coarse grid) use this
/// to turn an O(depth) walk into a walk bounded by how much finer `point`'s leaf is than `start`.
pub(crate) fn leaf_containing_from<
    const D: usize,
    const L: usize,
    const M: usize,
    const N: usize,
    T,
    U,
    V,
>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
    start: usize,
    point: &[usize; D],
) -> usize
where
    T: Cell,
    U: Slot,
{
    let mut index = start;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half: usize = tree.nodes[index].length.split().cells();
                let child = (0..D).fold(0, |acc, a| {
                    let mid: usize = corner[a].cells() + half;
                    acc | (usize::from(point[a] >= mid) << a)
                });
                index = orthants[child].slot();
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
    T: Cell,
    U: Slot,
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
                index = orthants[child].slot();
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
    T: Cell,
    U: Slot,
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
                let length: Scalar = leaf.length.scalar();
                let center = from_fn(|i| {
                    let c: Scalar = leaf.corner[i].scalar();
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
