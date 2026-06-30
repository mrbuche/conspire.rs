use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 6] = [
    [2, 1, 5, 5, 10, 0, 5, 7, 10, 2, 5],
    [2, 4, 1, 1, 5, 4, 0, 3, 15, 6, 0],
    [1, 3, 5, 7, 15, 1, 15, 6, 10, 0, 5],
    [1, 4, 3, 3, 15, 5, 5, 2, 15, 4, 0],
    [5, 1, 3, 7, 15, 4, 15, 3, 10, 0, 5],
    [5, 2, 1, 5, 15, 6, 10, 1, 15, 2, 0],
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
        [true, false, true, false, true, false, true],
        false,
    )
}
