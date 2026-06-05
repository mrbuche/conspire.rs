use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 8] = [
    [2, 1, 5, 5, 4, 0, 5, 15, 6, 0, 3],
    [2, 4, 1, 1, 5, 4, 0, 5, 7, 10, 2],
    [2, 0, 4, 0, 1, 5, 0, 0, 3, 15, 6],
    [2, 5, 0, 4, 0, 1, 10, 10, 2, 5, 7],
    [3, 0, 5, 6, 7, 3, 10, 10, 5, 5, 0],
    [3, 4, 0, 2, 6, 7, 5, 0, 4, 15, 1],
    [3, 1, 4, 3, 2, 6, 15, 5, 0, 10, 5],
    [3, 5, 1, 7, 3, 2, 15, 15, 1, 0, 4],
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
        [false, false, true, true, false, true, false],
        false,
    )
}
