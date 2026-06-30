use super::super::{D, M, N};
use super::three_face;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 12] = [
    [1, 5, 3, 7, 6, 2, 3, 15, 4, 0, 1],
    [5, 0, 3, 6, 2, 3, 7, 10, 0, 0, 5],
    [3, 5, 0, 6, 4, 0, 2, 15, 5, 5, 3],
    [5, 2, 0, 4, 0, 2, 6, 10, 1, 5, 7],
    [0, 5, 2, 4, 5, 1, 0, 10, 7, 15, 2],
    [5, 1, 2, 5, 1, 0, 4, 15, 3, 5, 6],
    [2, 5, 1, 5, 7, 3, 1, 10, 6, 10, 0],
    [5, 3, 1, 7, 3, 1, 5, 15, 2, 0, 4],
    [1, 3, 4, 3, 2, 0, 1, 15, 6, 10, 5],
    [3, 0, 4, 2, 0, 1, 3, 10, 4, 10, 7],
    [1, 2, 5, 5, 4, 6, 7, 5, 0, 0, 3],
    [2, 0, 5, 4, 6, 7, 5, 0, 2, 5, 1],
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
    three_face(
        tree,
        node,
        cell_subcells,
        center_nodes,
        data,
        [false, false, false, true, false, true, false],
        false,
    )
}
