use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 4] = [
    [2, 1, 5, 5, 10, 0, 5, 15, 10, 2, 5],
    [1, 3, 5, 7, 15, 5, 15, 15, 10, 0, 5],
    [3, 0, 5, 6, 15, 5, 10, 10, 15, 1, 0],
    [0, 2, 5, 4, 10, 0, 0, 10, 15, 3, 0],
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
        [true, true, true, true, true, false, true],
        false,
    )
}
