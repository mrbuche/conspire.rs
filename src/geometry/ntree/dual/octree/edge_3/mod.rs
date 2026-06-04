use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, N},
            },
        },
    },
    math::{Scalar, TensorVec},
};

type Edge = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
    [Scalar; D],
);

const EDGES: [Edge; 12] = [
    (0, 1, 2, 4, 2, 4, 3, 5, false, [0.0, 1.0, 0.0]),
    (0, 2, 0, 4, 1, 4, 3, 6, true, [1.0, 0.0, 0.0]),
    (0, 4, 0, 2, 1, 2, 5, 6, false, [1.0, 0.0, 0.0]),
    (1, 3, 1, 4, 0, 5, 2, 7, false, [-1.0, 0.0, 0.0]),
    (1, 5, 2, 1, 3, 0, 7, 4, false, [0.0, 1.0, 0.0]),
    (2, 3, 3, 4, 0, 6, 1, 7, true, [0.0, -1.0, 0.0]),
    (2, 6, 3, 0, 0, 3, 4, 7, false, [0.0, -1.0, 0.0]),
    (3, 7, 1, 3, 2, 1, 6, 5, false, [-1.0, 0.0, 0.0]),
    (4, 5, 5, 2, 0, 6, 1, 7, false, [0.0, 0.0, -1.0]),
    (4, 6, 5, 0, 0, 5, 2, 7, true, [0.0, 0.0, -1.0]),
    (5, 7, 5, 1, 1, 4, 3, 6, false, [0.0, 0.0, -1.0]),
    (6, 7, 5, 3, 2, 4, 3, 5, true, [0.0, 0.0, -1.0]),
];

pub fn edge_transition_3<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    for node in tree.iter().filter(|node| node.is_tree()) {
        let cell_subnodes = tree.leaves(node);
        for &edge in EDGES.iter() {
            template(
                edge,
                &cell_subnodes,
                center_nodes,
                nodes_map,
                node_index,
                tree,
                connectivity,
                coordinates,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    edge: Edge,
    cell_subnodes: &[Option<U>; N],
    center_nodes: &[usize],
    nodes_map: &mut NodeMap<D>,
    node_index: &mut usize,
    tree: &Octree<T, U>,
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let (subcell_a, subcell_b, facet_m, facet_n, c, d, e, f, flip, direction) = edge;
    if let Some(node_a) = cell_subnodes[subcell_a]
        && let Some(node_b) = cell_subnodes[subcell_b]
        && let Some(a_m) = tree.nodes[node_a.into()].facets[facet_m]
        && let Some(a_n) = tree.nodes[node_a.into()].facets[facet_n]
        && let Some(b_m) = tree.nodes[node_b.into()].facets[facet_m]
        && let Some(b_n) = tree.nodes[node_b.into()].facets[facet_n]
    {
        let a_m_leaves = tree.leaves(&tree.nodes[a_m.into()]);
        let a_n_leaves = tree.leaves(&tree.nodes[a_n.into()]);
        let b_m_leaves = tree.leaves(&tree.nodes[b_m.into()]);
        let b_n_leaves = tree.leaves(&tree.nodes[b_n.into()]);
        if let Some(a_m_c) = a_m_leaves[c]
            && let Some(a_m_e) = a_m_leaves[e]
            && let Some(a_n_d) = a_n_leaves[d]
            && let Some(a_n_f) = a_n_leaves[f]
            && let Some(b_m_c) = b_m_leaves[c]
            && let Some(b_m_e) = b_m_leaves[e]
            && let Some(b_n_d) = b_n_leaves[d]
            && let Some(b_n_f) = b_n_leaves[f]
            && let Some(diagonal_a) = tree.nodes[a_m_c.into()].facets[facet_n]
            && tree.nodes[diagonal_a.into()].is_leaf()
            && let Some(subdiagonal_a) = tree.nodes[a_m_e.into()].facets[facet_n]
            && tree.nodes[subdiagonal_a.into()].is_leaf()
            && let Some(diagonal_b) = tree.nodes[b_m_e.into()].facets[facet_n]
            && tree.nodes[diagonal_b.into()].is_leaf()
            && let Some(subdiagonal_b) = tree.nodes[b_m_c.into()].facets[facet_n]
            && tree.nodes[subdiagonal_b.into()].is_leaf()
        {
            let length: Scalar = tree.nodes[a_m_e.into()].length.into();
            let offset = &Coordinate::const_from(direction) * length;
            let base_0 = coordinates[center_nodes[a_m_e.into()]].clone();
            let base_1 = coordinates[center_nodes[b_m_c.into()]].clone();
            coordinates.push(&base_0 + &offset);
            coordinates.push(&base_1 + &offset);
            let new = *node_index;
            for k in 0..2 {
                let coordinate = &coordinates[new + k];
                let key = [
                    (2.0 * coordinate[0]) as usize,
                    (2.0 * coordinate[1]) as usize,
                    (2.0 * coordinate[2]) as usize,
                ];
                assert!(
                    nodes_map.insert(key, new + k).is_none(),
                    "edge_3 duplicate node at {key:?}"
                );
            }
            *node_index += 2;
            let center_a = center_nodes[node_a.into()];
            let center_b = center_nodes[node_b.into()];
            let a_m_c = center_nodes[a_m_c.into()];
            let a_m_e = center_nodes[a_m_e.into()];
            let a_n_d = center_nodes[a_n_d.into()];
            let a_n_f = center_nodes[a_n_f.into()];
            let b_m_c = center_nodes[b_m_c.into()];
            let b_m_e = center_nodes[b_m_e.into()];
            let b_n_d = center_nodes[b_n_d.into()];
            let b_n_f = center_nodes[b_n_f.into()];
            let diag_a = center_nodes[diagonal_a.into()];
            let subdiag_a = center_nodes[subdiagonal_a.into()];
            let diag_b = center_nodes[diagonal_b.into()];
            let subdiag_b = center_nodes[subdiagonal_b.into()];
            if flip {
                connectivity.push([new, a_m_e, subdiag_a, a_n_f, center_a, a_m_c, diag_a, a_n_d]);
                connectivity.push([
                    new + 1,
                    b_m_c,
                    subdiag_b,
                    b_n_d,
                    new,
                    a_m_e,
                    subdiag_a,
                    a_n_f,
                ]);
                connectivity.push([
                    center_b,
                    b_m_e,
                    diag_b,
                    b_n_f,
                    new + 1,
                    b_m_c,
                    subdiag_b,
                    b_n_d,
                ]);
            } else {
                connectivity.push([center_a, a_m_c, diag_a, a_n_d, new, a_m_e, subdiag_a, a_n_f]);
                connectivity.push([
                    new,
                    a_m_e,
                    subdiag_a,
                    a_n_f,
                    new + 1,
                    b_m_c,
                    subdiag_b,
                    b_n_d,
                ]);
                connectivity.push([
                    new + 1,
                    b_m_c,
                    subdiag_b,
                    b_n_d,
                    center_b,
                    b_m_e,
                    diag_b,
                    b_n_f,
                ]);
            }
        }
    }
}
