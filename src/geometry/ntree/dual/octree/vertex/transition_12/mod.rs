use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 12] = [
    [2, 1, 5, 5, 10, 0, 5, 15, 10, 10, 5],
    [2, 4, 1, 1, 5, 4, 0, 5, 15, 15, 0],
    [1, 3, 5, 7, 15, 1, 15, 15, 10, 0, 5],
    [1, 4, 3, 3, 15, 5, 5, 5, 15, 10, 0],
    [3, 0, 5, 6, 15, 3, 10, 10, 15, 5, 0],
    [3, 4, 0, 2, 10, 7, 5, 0, 10, 10, 5],
    [0, 2, 5, 4, 10, 2, 0, 10, 15, 15, 0],
    [0, 4, 2, 0, 0, 6, 0, 0, 10, 15, 5],
    [4, 1, 2, 1, 0, 2, 5, 5, 0, 10, 15],
    [4, 3, 1, 3, 5, 0, 5, 15, 5, 10, 10],
    [5, 1, 3, 7, 15, 4, 15, 15, 10, 0, 5],
    [5, 2, 1, 5, 15, 6, 10, 5, 15, 5, 0],
];

pub fn template<T, U>(
    tree: &Octree<T, U>,
    node: &Node<D, M, N, T, U>,
    cell_subcells: &[U; N],
    center_nodes: &[usize],
    data: [usize; 11],
) -> Option<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    face_plus_two(
        tree,
        node,
        cell_subcells,
        center_nodes,
        data,
        [true, false, true, true, true, true, true],
        true,
    )
}
