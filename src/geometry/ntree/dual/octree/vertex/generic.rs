use super::super::{D, N};
use crate::geometry::ntree::{
    Octree,
    node::{Kind, split::Split},
};
use std::{array::from_fn, ops::Add};

/// Generic vertex dual: one cell-center hex per interior octree vertex, built by
/// "descend toward V" instead of the 21 hand-coded templates.
///
/// Every interior vertex `V` is the `(+,+,+)` corner of exactly one leaf (its `(-,-,-)`
/// octant), so iterating leaves enumerates each interior `V` once. For octant `d` the
/// leaf touching `V` is found by [`find_leaf_octant`] (a point descent toward `V`), and
/// the eight centers are wound `[0,1,3,2,4,5,7,6]` (octant index -> hex node), matching
/// `uniform_transition_1`.
pub(crate) fn vertex_dual_generic<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize> + Add<Output = T> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];
    let root = &tree.nodes[0];
    let lo = root.corner;
    let hi: [T; D] = from_fn(|a| root.corner[a] + root.length);
    let mut hexes = Vec::new();
    for node in tree.iter().filter(|node| node.is_leaf()) {
        let v: [T; D] = from_fn(|a| node.corner[a] + node.length);
        if (0..D).all(|a| lo[a] < v[a] && v[a] < hi[a]) {
            let cells: [usize; N] = from_fn(|d| find_leaf_octant(tree, &v, d));
            hexes.push(from_fn(|k| center_nodes[cells[WIND[k]]]));
        }
    }
    hexes
}

/// The leaf touching vertex `v` from octant `d`. Descends from the root toward `v`; when
/// `v` lands exactly on a child split plane (always, since `v` is a grid corner), the tie
/// is broken toward octant `d` so the descent enters the cell on `d`'s side of `v`.
fn find_leaf_octant<T, U>(tree: &Octree<T, U>, v: &[T; D], d: usize) -> usize
where
    T: Copy + Add<Output = T> + PartialOrd + Split,
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
                    let bit = if v[a] > mid {
                        1
                    } else if v[a] < mid {
                        0
                    } else {
                        (d >> a) & 1
                    };
                    acc | (bit << a)
                });
                index = orthants[child].into();
            }
        }
    }
}
