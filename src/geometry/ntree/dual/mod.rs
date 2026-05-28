pub mod quadtree;
pub mod octree;

use crate::geometry::{mesh::PrimitiveMesh, ntree::Orthotree};
use std::array::from_fn;

pub trait Dualization<const D: usize, const I: usize, const M: usize, const N: usize, T> {
    fn dualize(&mut self) -> PrimitiveMesh<D, I, M, N, T>;
}

fn uniform_transition_1<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U>,
    center_nodes: &[V],
    connectivity: &mut Vec<[V; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy,
{
    let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
    connectivity.extend(
        tree.iter()
            .filter_map(|node| tree.all_leaves(node))
            .map(|leaves| {
                from_fn(|i| {
                    let face = i & face_mask;
                    let vertex = (i & !face_mask) | (face ^ (face >> 1));
                    center_nodes[leaves[vertex].into()]
                })
            }),
    )
}
