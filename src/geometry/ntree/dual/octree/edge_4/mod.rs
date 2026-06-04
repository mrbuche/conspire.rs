use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, N, facet_direction},
            },
        },
    },
    math::Scalar,
};

type Edge = (usize, usize, usize, usize, usize, usize, usize, usize);

const EDGES: [Edge; 6] = [
    (4, 6, 0, 5, 7, 5, 2, 0),
    (5, 7, 5, 1, 3, 1, 6, 4),
    (5, 4, 2, 5, 6, 7, 0, 1),
    (7, 6, 5, 3, 2, 3, 4, 5),
    (6, 2, 0, 3, 3, 7, 0, 4),
    (7, 3, 3, 1, 1, 5, 2, 6),
];

pub fn edge_transition_4<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    for node in tree.iter() {
        if let Some(cell_subcells) = tree.all_leaves(node) {
            for &edge in EDGES.iter() {
                template(
                    edge,
                    cell_subcells,
                    center_nodes,
                    coordinates,
                    connectivity,
                    nodes_map,
                    tree,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    edge: Edge,
    cell_subcells: &[U; N],
    center_nodes: &[usize],
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
    tree: &Octree<T, U>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let (subcell_a, subcell_b, facet_m, facet_n, m_p, m_q, n_p, n_q) = edge;
    let subcell_a_node: usize = cell_subcells[subcell_a].into();
    let subcell_b_node: usize = cell_subcells[subcell_b].into();
    if let Some(a_m) = tree.nodes[subcell_a_node].facets[facet_m]
        && let Some(a_n) = tree.nodes[subcell_a_node].facets[facet_n]
        && let Some(diagonal_a) = tree.nodes[a_m.into()].facets[facet_n]
        && tree.nodes[diagonal_a.into()].is_leaf()
        && let Some(b_m) = tree.nodes[subcell_b_node].facets[facet_m]
        && let Some(b_n) = tree.nodes[subcell_b_node].facets[facet_n]
        && let Some(diagonal_b) = tree.nodes[b_m.into()].facets[facet_n]
        && tree.nodes[diagonal_b.into()].is_leaf()
        && let Some(a_m_leaves) = tree.all_leaves(&tree.nodes[a_m.into()])
        && let Some(a_n_leaves) = tree.all_leaves(&tree.nodes[a_n.into()])
        && let Some(b_m_leaves) = tree.all_leaves(&tree.nodes[b_m.into()])
        && let Some(b_n_leaves) = tree.all_leaves(&tree.nodes[b_n.into()])
    {
        let length: Scalar = tree.nodes[a_m_leaves[m_p].into()].length.into();
        let offset_m = &facet_direction(facet_m) * length;
        let offset_n = &facet_direction(facet_n) * length;
        let a_m_p = center_nodes[a_m_leaves[m_p].into()];
        let b_m_q = center_nodes[b_m_leaves[m_q].into()];
        let find = |coordinate: Coordinate<D>| -> Option<usize> {
            nodes_map
                .get(&[
                    (2.0 * coordinate[0]) as usize,
                    (2.0 * coordinate[1]) as usize,
                    (2.0 * coordinate[2]) as usize,
                ])
                .copied()
        };
        if let Some(node_1) = find(&coordinates[a_m_p] - &offset_m)
            && let Some(node_2) = find(&coordinates[b_m_q] - &offset_m)
            && let Some(node_3) = find(&coordinates[a_m_p] + &offset_n)
            && let Some(node_4) = find(&coordinates[b_m_q] + &offset_n)
        {
            let center_a = center_nodes[subcell_a_node];
            let center_b = center_nodes[subcell_b_node];
            let a_m_q = center_nodes[a_m_leaves[m_q].into()];
            let b_m_p = center_nodes[b_m_leaves[m_p].into()];
            let a_n_p = center_nodes[a_n_leaves[n_p].into()];
            let a_n_q = center_nodes[a_n_leaves[n_q].into()];
            let b_n_p = center_nodes[b_n_leaves[n_p].into()];
            let b_n_q = center_nodes[b_n_leaves[n_q].into()];
            let diag_a = center_nodes[diagonal_a.into()];
            let diag_b = center_nodes[diagonal_b.into()];
            connectivity.push([a_m_p, node_1, node_2, b_m_q, node_3, a_n_p, b_n_q, node_4]);
            connectivity.push([b_m_q, node_2, center_b, b_m_p, node_4, b_n_q, b_n_p, diag_b]);
            connectivity.push([a_m_q, center_a, node_1, a_m_p, diag_a, a_n_q, a_n_p, node_3]);
        }
    }
}
