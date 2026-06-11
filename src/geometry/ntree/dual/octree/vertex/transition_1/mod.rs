use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 6] = [
    [2, 1, 5, 5, 4, 0, 1, 15, 10, 0, 5],
    [1, 3, 5, 7, 5, 1, 3, 15, 10, 0, 5],
    [3, 0, 5, 6, 7, 3, 2, 10, 15, 5, 0],
    [0, 2, 5, 4, 6, 2, 0, 10, 15, 5, 0],
    [4, 2, 0, 0, 2, 3, 1, 0, 10, 15, 5],
    [5, 2, 1, 5, 7, 6, 4, 5, 15, 10, 0],
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
        [false, false, false, true, true, true, true],
        false,
    )
}
