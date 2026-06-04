use super::super::{D, M, N};
use super::sub_subnode;
use crate::geometry::ntree::{Octree, node::Node};

// (O, A, AB, B) => (o, a, ab, b): the cell and its `facet_a`/`facet_b`/diagonal
// neighbors are coarse (a subcell each); across `facet` each meets a fine cell.
// [facet, facet_a, facet_b, subcell, subcell_a, subcell_b, subcell_ab,
//  sub_subcell, sub_subcell_a, sub_subcell_b, sub_subcell_ab].
pub const DATA: [[usize; 11]; 6] = [
    [2, 1, 5, 5, 4, 1, 0, 15, 10, 5, 0],
    [1, 3, 5, 7, 5, 3, 1, 15, 10, 5, 0],
    [3, 0, 5, 6, 7, 2, 3, 10, 15, 0, 5],
    [0, 2, 5, 4, 6, 0, 2, 10, 15, 0, 5],
    [4, 2, 0, 0, 2, 1, 3, 0, 10, 5, 15],
    [5, 2, 1, 5, 7, 4, 6, 5, 15, 0, 10],
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
        facet,
        facet_a,
        facet_b,
        sc,
        sc_a,
        sc_b,
        sc_ab,
        ss,
        ss_a,
        ss_b,
        ss_ab,
    ] = data;
    let cell_a = node.facets[facet_a]?;
    let cell_b = node.facets[facet_b]?;
    let cell_ab = tree.nodes[cell_a.into()].facets[facet_b]?;
    let cell_a_subcells = tree.all_leaves(&tree.nodes[cell_a.into()])?;
    let cell_b_subcells = tree.all_leaves(&tree.nodes[cell_b.into()])?;
    let cell_ab_subcells = tree.all_leaves(&tree.nodes[cell_ab.into()])?;
    let face = sub_subnode(tree, node.facets[facet]?, facet, ss)?;
    let face_a = sub_subnode(tree, tree.nodes[cell_a.into()].facets[facet]?, facet, ss_a)?;
    let face_b = sub_subnode(tree, tree.nodes[cell_b.into()].facets[facet]?, facet, ss_b)?;
    let face_ab = sub_subnode(
        tree,
        tree.nodes[cell_ab.into()].facets[facet]?,
        facet,
        ss_ab,
    )?;
    Some([
        center_nodes[cell_subcells[sc].into()],
        center_nodes[cell_a_subcells[sc_a].into()],
        center_nodes[cell_ab_subcells[sc_ab].into()],
        center_nodes[cell_b_subcells[sc_b].into()],
        center_nodes[face.into()],
        center_nodes[face_a.into()],
        center_nodes[face_ab.into()],
        center_nodes[face_b.into()],
    ])
}
