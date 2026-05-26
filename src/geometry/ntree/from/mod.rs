use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Orthotree,
            node::{Kind, Node},
            subdivide::Pairing,
        },
    },
    math::TensorVec,
};
use std::array::from_fn;

impl<const D: usize, const M: usize, const N: usize, const I: usize>
    From<(Coordinates<D, I>, f64)> for Orthotree<D, M, N, u16, usize>
{
    fn from((coordinates, min_length): (Coordinates<D, I>, f64)) -> Self {
        if coordinates.is_empty() {
            return Self {
                nodes: vec![Node {
                    corner: [0u16; D],
                    length: 1,
                    facets: [usize::MAX; M],
                    kind: Kind::Leaf(()),
                }],
            };
        }
        let mut min_coord: [f64; D] = from_fn(|_| f64::INFINITY);
        let mut max_coord: [f64; D] = from_fn(|_| f64::NEG_INFINITY);
        for point in &coordinates {
            for ax in 0..D {
                min_coord[ax] = min_coord[ax].min(point[ax]);
                max_coord[ax] = max_coord[ax].max(point[ax]);
            }
        }
        let max_extent = (0..D)
            .map(|ax| max_coord[ax] - min_coord[ax])
            .fold(0.0f64, f64::max);
        let levels = if max_extent <= 0.0 {
            0u32
        } else {
            (max_extent / min_length).log2().ceil().max(0.0) as u32
        };
        let root_length: u16 = 1u16.checked_shl(levels).unwrap_or(u16::MAX);
        let center: [f64; D] = from_fn(|ax| (min_coord[ax] + max_coord[ax]) / 2.0);
        let mut tree = Self {
            nodes: vec![Node {
                corner: [0u16; D],
                length: root_length,
                facets: [usize::MAX; M],
                kind: Kind::Leaf(()),
            }],
        };
        for point in &coordinates {
            let int_coord: [u16; D] = from_fn(|ax| {
                let v = ((point[ax] - center[ax]) / min_length
                    + root_length as f64 / 2.0)
                    .floor() as i64;
                v.clamp(0, root_length as i64 - 1) as u16
            });
            loop {
                let index = find_leaf(&tree, &int_coord);
                if tree.nodes[index].length <= 1 {
                    break;
                }
                tree.subdivide(index, Pairing::Regular).ok();
            }
        }
        tree
    }
}

fn find_leaf<const D: usize, const M: usize, const N: usize, U>(
    tree: &Orthotree<D, M, N, u16, U>,
    coord: &[u16; D],
) -> usize {
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half = tree.nodes[index].length / 2;
                let child_i = (0..D).fold(0, |acc, ax| {
                    if coord[ax] >= corner[ax] + half {
                        acc | (1 << ax)
                    } else {
                        acc
                    }
                });
                index = orthants[child_i].into();
            }
        }
    }
}
