use crate::geometry::ntree::{
    Octree,
    dual::{
        NodeMap, incident_leaf,
        octree::{D, N},
    },
    node::{Kind, split::Split},
};
use std::{array::from_fn, ops::Add};

const L2: usize = 4;

const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];

// The star hexes themselves come from the generic Star rule in
// dual/mod.rs; this pass fills the one weak configuration the star
// skips, which needs two elements: see cap() below.
pub fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
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
            if distinct.len() != N {
                continue;
            }
            let lengths: [usize; N] = from_fn(|o| tree.nodes[cells[o]].length.into());
            let shortest = *lengths.iter().min().unwrap();
            let longest = *lengths.iter().max().unwrap();
            let coordinate: [usize; D] = from_fn(|a| vertex[a].into());
            if longest == 4 * shortest
                && !(0..D).all(|a| coordinate[a].is_multiple_of(2 * longest))
            {
                cap(
                    tree,
                    center_nodes,
                    &coordinate,
                    &cells,
                    &lengths,
                    shortest,
                    connectivity,
                    nodes_map,
                );
            }
        }
    }
}

// A 4:1 vertex at an odd multiple of the longest length, where the two
// longest cells are siblings pairing across the vertex plane. The weak
// edge tube on the finest side ends there with a Steiner-cornered ring,
// so the fill is two elements: the star hex with the paired cells'
// corners moved to the Steiner nodes the face transitions planted just
// inside them, plus one connector hex bridging from that face of the
// star to the next cells and the paired cells' centers on the side away
// from the finest cells.
#[allow(clippy::too_many_arguments)]
fn cap<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    vertex: &[usize; D],
    cells: &[usize; N],
    lengths: &[usize; N],
    shortest: usize,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let longest = 4 * shortest;
    let odd: Vec<usize> = (0..D)
        .filter(|&a| !vertex[a].is_multiple_of(2 * longest))
        .collect();
    let [axis] = odd[..] else {
        return;
    };
    let paired: Vec<usize> = (0..N).filter(|&o| lengths[o] == longest).collect();
    let [big_lo, big_hi] = paired[..] else {
        return;
    };
    if big_lo ^ big_hi != 1 << axis {
        return;
    }
    let smalls: Vec<usize> = (0..N).filter(|&o| lengths[o] == shortest).collect();
    let side = (smalls[0] >> axis) & 1;
    if smalls.iter().any(|&o| (o >> axis) & 1 != side) {
        return;
    }
    let star: [Option<usize>; N] = from_fn(|o| {
        if lengths[o] == longest {
            let key: [usize; D] = from_fn(|a| {
                if (o >> a) & 1 == 1 {
                    2 * vertex[a] + 2 * shortest
                } else {
                    2 * vertex[a] - 2 * shortest
                }
            });
            nodes_map.get(&key).copied()
        } else {
            Some(center_nodes[cells[o]])
        }
    });
    if star.iter().any(|corner| corner.is_none()) {
        return;
    }
    let cap_side = 1 - side;
    let slot = |t: usize| {
        let u = (axis + 1) % D;
        let w = (axis + 2) % D;
        (cap_side << axis) | ((t & 1) << u) | (((t >> 1) & 1) << w)
    };
    let outer: [Option<usize>; L2] = from_fn(|t| {
        let o = slot(t);
        if lengths[o] == longest {
            Some(center_nodes[cells[o]])
        } else {
            let point: [usize; D] = from_fn(|a| {
                let delta = if a == axis { 3 * shortest } else { shortest };
                if (o >> a) & 1 == 1 {
                    vertex[a] + delta
                } else {
                    vertex[a] - delta
                }
            });
            let cell = leaf_at(tree, &point, o);
            let length: usize = tree.nodes[cell].length.into();
            (length == 2 * shortest).then(|| center_nodes[cell])
        }
    });
    if outer.iter().any(|corner| corner.is_none()) {
        return;
    }
    connectivity.push(from_fn(|k| star[WIND[k]].unwrap()));
    let inner: [usize; L2] = from_fn(|t| star[slot(t)].unwrap());
    let outer: [usize; L2] = from_fn(|t| outer[t].unwrap());
    let (bottom, top) = if cap_side == 1 {
        (inner, outer)
    } else {
        (outer, inner)
    };
    const QUAD: [usize; L2] = [0, 1, 3, 2];
    connectivity.push(from_fn(|k| {
        if k < L2 {
            bottom[QUAD[k]]
        } else {
            top[QUAD[k - L2]]
        }
    }));
}

fn leaf_at<T, U>(tree: &Octree<T, U>, point: &[usize; D], direction: usize) -> usize
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let node = &tree.nodes[index];
                let length: usize = node.length.into();
                let child = (0..D).fold(0, |acc, a| {
                    let mid: usize = node.corner[a].into() + length / 2;
                    let bit = if point[a] > mid {
                        1
                    } else if point[a] < mid {
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
