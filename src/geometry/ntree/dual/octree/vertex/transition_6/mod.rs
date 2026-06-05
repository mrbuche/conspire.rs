use super::super::{D, M, N};
use super::three_face;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 7] = [
    [1, 5, 3, 7, 6, 2, 15, 15, 4, 0, 5],
    [3, 0, 4, 2, 0, 1, 5, 10, 4, 10, 15],
    [3, 5, 0, 6, 4, 0, 10, 15, 5, 5, 15],
    [2, 5, 1, 5, 7, 3, 5, 10, 6, 10, 0],
    [3, 1, 5, 7, 5, 4, 15, 15, 1, 0, 5],
    [0, 4, 3, 2, 3, 7, 10, 0, 1, 5, 0],
    [1, 3, 4, 3, 2, 0, 5, 15, 6, 10, 15],
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
        [false, false, true, true, false, true, true],
        false,
    )
}
