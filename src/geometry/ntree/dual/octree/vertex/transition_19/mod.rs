use super::super::{D, M, N};
use super::face_plus_two;
use crate::geometry::ntree::{Octree, node::Node};

pub const DATA: [[usize; 11]; 4] = [
    [2, 1, 5, 5, 4, 0, 5, 7, 10, 2, 3],
    [2, 4, 1, 1, 5, 4, 0, 3, 15, 6, 2],
    [2, 0, 4, 0, 1, 5, 0, 2, 5, 7, 6],
    [2, 5, 0, 4, 0, 1, 10, 6, 0, 3, 7],
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
        [false, false, true, false, true, false, false],
        false,
    )
}
