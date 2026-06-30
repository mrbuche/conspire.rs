use super::super::{D, M, N};
use super::three_face;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 8] = [
    [1, 5, 3, 7, 15, 10, 15, 15, 10, 0, 5],
    [3, 5, 0, 6, 10, 0, 10, 15, 15, 5, 15],
    [0, 5, 2, 4, 10, 5, 0, 10, 15, 15, 10],
    [2, 5, 1, 5, 15, 15, 5, 10, 10, 10, 0],
    [1, 4, 2, 1, 0, 0, 5, 5, 5, 10, 15],
    [2, 4, 0, 0, 0, 10, 0, 0, 5, 15, 5],
    [0, 4, 3, 2, 5, 15, 10, 0, 0, 5, 0],
    [3, 4, 1, 3, 5, 5, 15, 5, 0, 0, 10],
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
        [true, true, true, true, true, true, true],
        false,
    )
}
