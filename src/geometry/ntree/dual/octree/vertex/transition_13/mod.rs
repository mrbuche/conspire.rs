use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 8] = [
    [2, 1, 5, 5, 10, 0, 5, 15, 6, 10, 5],
    [2, 4, 1, 1, 5, 4, 0, 5, 7, 15, 0],
    [1, 2, 4, 1, 5, 7, 5, 0, 2, 10, 10],
    [1, 4, 3, 3, 15, 5, 5, 5, 6, 10, 0],
    [3, 1, 4, 3, 5, 6, 15, 5, 0, 0, 15],
    [3, 4, 0, 2, 10, 7, 5, 0, 4, 10, 5],
    [4, 1, 2, 1, 0, 2, 5, 5, 4, 10, 15],
    [4, 3, 1, 3, 5, 0, 5, 15, 5, 10, 10],
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
        [true, false, true, true, false, true, true],
        true,
    )
}
