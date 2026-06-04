use super::super::{D, M, N};
use super::sub_subnode;
use crate::geometry::ntree::{Octree, node::Node};

// (O, a, ab, b) => (O, a, ab, b): the cell and its `facet_c` neighbor are coarse (a
// subcell each); the `facet_a`/`facet_b`/diagonal neighbors of both are fine.
// [facet_a, facet_b, facet_c, subcell, subcell_c, sub_subcell_a, sub_subcell_b,
//  sub_subcell_ab, sub_subcell_c_a, sub_subcell_c_b, sub_subcell_c_ab].
pub const DATA: [[usize; 11]; 12] = [
    [2, 0, 5, 4, 0, 10, 10, 15, 0, 0, 5],
    [1, 2, 5, 5, 1, 10, 15, 10, 0, 5, 0],
    [3, 1, 5, 7, 3, 15, 15, 10, 5, 5, 0],
    [0, 3, 5, 6, 2, 15, 10, 15, 5, 0, 5],
    [4, 1, 3, 3, 1, 15, 5, 15, 5, 0, 10],
    [3, 4, 1, 3, 2, 5, 15, 5, 0, 10, 0],
    [0, 4, 3, 2, 0, 5, 10, 15, 0, 0, 5],
    [2, 4, 0, 0, 1, 0, 0, 10, 5, 5, 15],
    [5, 3, 1, 7, 6, 15, 15, 5, 10, 10, 0],
    [2, 5, 1, 5, 4, 15, 5, 15, 10, 0, 10],
    [5, 0, 3, 6, 4, 10, 15, 5, 0, 10, 0],
    [1, 5, 3, 7, 5, 15, 15, 10, 10, 5, 0],
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
    let [
        facet_a,
        facet_b,
        facet_c,
        sc,
        sc_c,
        ss_a,
        ss_b,
        ss_ab,
        ss_c_a,
        ss_c_b,
        ss_c_ab,
    ] = data;
    let cell_a = node.facets[facet_a]?;
    let cell_b = node.facets[facet_b]?;
    let cell_ab = tree.nodes[cell_a.into()].facets[facet_b]?;
    let cell_c = node.facets[facet_c]?;
    let cell_c_subcells = tree.all_leaves(&tree.nodes[cell_c.into()])?;
    let cell_c_a = tree.nodes[cell_c.into()].facets[facet_a]?;
    let cell_c_b = tree.nodes[cell_c.into()].facets[facet_b]?;
    let cell_c_ab = tree.nodes[cell_c_a.into()].facets[facet_b]?;
    let face_a = sub_subnode(tree, cell_a, facet_a, ss_a)?;
    let face_b = sub_subnode(tree, cell_b, facet_b, ss_b)?;
    let face_ab = sub_subnode(tree, cell_ab, facet_b, ss_ab)?;
    let face_c_a = sub_subnode(tree, cell_c_a, facet_a, ss_c_a)?;
    let face_c_b = sub_subnode(tree, cell_c_b, facet_b, ss_c_b)?;
    let face_c_ab = sub_subnode(tree, cell_c_ab, facet_b, ss_c_ab)?;
    Some([
        center_nodes[cell_subcells[sc].into()],
        center_nodes[face_b.into()],
        center_nodes[face_ab.into()],
        center_nodes[face_a.into()],
        center_nodes[cell_c_subcells[sc_c].into()],
        center_nodes[face_c_b.into()],
        center_nodes[face_c_ab.into()],
        center_nodes[face_c_a.into()],
    ])
}
