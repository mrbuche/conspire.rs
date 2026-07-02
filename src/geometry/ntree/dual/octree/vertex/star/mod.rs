use crate::geometry::ntree::{
    Octree,
    dual::octree::{D, N},
    node::{Kind, split::Split},
};
use std::{array::from_fn, ops::Add};

const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];

// Weak (face-balanced only) vertex star: at a parent-grid vertex whose eight
// incident cells are distinct and span a 4:1 size ratio, the dual element is
// the hex with those cells' centers as corners. This covers every weak
// vertex configuration; vertices interior to a parent are filled by the
// Steiner-cornered edge transition_5 pair templates instead.
pub fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Add<Output = T> + Copy + PartialOrd + Split + Into<usize>,
    U: Copy + Into<usize>,
{
    let root = &tree.nodes[0];
    let lo = root.corner;
    let hi: [T; D] = from_fn(|a| root.corner[a] + root.length);
    for node in tree.iter().filter(|node| node.is_leaf()) {
        let vertex: [T; D] = from_fn(|a| node.corner[a] + node.length);
        if (0..D).all(|a| lo[a] < vertex[a] && vertex[a] < hi[a]) {
            let cells: [usize; N] = from_fn(|d| incident_leaf(tree, &vertex, d));
            let mut distinct = cells.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            let shortest = cells
                .iter()
                .map(|&cell| tree.nodes[cell].length.into())
                .min()
                .unwrap();
            let longest = cells
                .iter()
                .map(|&cell| tree.nodes[cell].length.into())
                .max()
                .unwrap();
            if distinct.len() == N
                && longest == 4 * shortest
                && (0..D).all(|a| {
                    let coordinate: usize = vertex[a].into();
                    coordinate.is_multiple_of(2 * longest)
                })
            {
                connectivity.push(from_fn(|k| center_nodes[cells[WIND[k]]]));
            }
        }
    }
}

fn incident_leaf<T, U>(tree: &Octree<T, U>, vertex: &[T; D], direction: usize) -> usize
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
