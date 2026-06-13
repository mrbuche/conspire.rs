use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, N},
            },
            node::Node,
        },
    },
    math::Scalar,
};

type Edge = (usize, usize, usize, usize, usize, usize, usize, usize);

// same edge layout as transition_3; the difference is the diagonal sits one
// further level of refinement down (is_tree instead of is_leaf)
const EDGES: [Edge; 12] = [
    (0, 1, 2, 4, 2, 4, 3, 5),
    (0, 2, 0, 4, 1, 4, 3, 6),
    (0, 4, 0, 2, 1, 2, 5, 6),
    (1, 3, 1, 4, 0, 5, 2, 7),
    (1, 5, 2, 1, 3, 0, 7, 4),
    (2, 3, 3, 4, 0, 6, 1, 7),
    (2, 6, 3, 0, 0, 3, 4, 7),
    (3, 7, 1, 3, 2, 1, 6, 5),
    (4, 5, 5, 2, 0, 6, 1, 7),
    (4, 6, 5, 0, 0, 5, 2, 7),
    (5, 7, 5, 1, 1, 4, 3, 6),
    (6, 7, 5, 3, 2, 4, 3, 5),
];

#[allow(clippy::too_many_arguments, unused_variables)]
pub fn template<T, U>(
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
            if identifies(edge, &cell_subnodes, tree) {
                // TODO: connectivity + coordinate creation
            }
        }
    }
}

#[allow(unused_variables)]
fn identifies<T, U>(edge: Edge, cell_subnodes: &[Option<U>; N], tree: &Octree<T, U>) -> bool
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let (subcell_a, subcell_b, facet_m, facet_n, c, d, e, f) = edge;
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
            && tree.nodes[diagonal_a.into()].is_tree()
            && let Some(subdiagonal_a) = tree.nodes[a_m_e.into()].facets[facet_n]
            && tree.nodes[subdiagonal_a.into()].is_tree()
            && let Some(diagonal_b) = tree.nodes[b_m_e.into()].facets[facet_n]
            && tree.nodes[diagonal_b.into()].is_tree()
            && let Some(subdiagonal_b) = tree.nodes[b_m_c.into()].facets[facet_n]
            && tree.nodes[subdiagonal_b.into()].is_tree()
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
pub(crate) fn count<T, U>(tree: &Octree<T, U>) -> usize
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter()
        .filter(|node| node.is_tree())
        .map(|node| {
            let cell_subnodes = tree.leaves(node);
            EDGES
                .iter()
                .filter(|&&edge| identifies(edge, &cell_subnodes, tree))
                .count()
        })
        .sum()
}
