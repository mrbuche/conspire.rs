use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, N},
            },
            node::Node,
        },
    },
    math::Scalar,
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

// same edge layout as transition_3; the diagonal sits one further level of
// refinement down (is_tree instead of is_leaf)
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
            template_inner(
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

#[allow(clippy::too_many_arguments, unused_variables)]
fn template_inner<T, U>(
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
            && tree.nodes[diagonal_a.into()].is_tree()
            && let Some(subdiagonal_a) = tree.nodes[a_m_e.into()].facets[facet_n]
            && tree.nodes[subdiagonal_a.into()].is_tree()
            && let Some(diagonal_b) = tree.nodes[b_m_e.into()].facets[facet_n]
            && tree.nodes[diagonal_b.into()].is_tree()
            && let Some(subdiagonal_b) = tree.nodes[b_m_c.into()].facets[facet_n]
            && tree.nodes[subdiagonal_b.into()].is_tree()
        {
            // descend each subdiagonal to its child at the inner corner nearest
            // the edge center (toward -facet_m, -facet_n, and the opposite subcell)
            let axis_m = facet_m >> 1;
            let axis_n = facet_n >> 1;
            let axis_t = 3 - axis_m - axis_n;
            let corner = ((1 - (facet_m & 1)) << axis_m) | ((1 - (facet_n & 1)) << axis_n);
            let orthant_a = corner | (((subcell_b >> axis_t) & 1) << axis_t);
            let orthant_b = corner | (((subcell_a >> axis_t) & 1) << axis_t);
            let sub_a = tree.nodes[subdiagonal_a.into()].orthants().unwrap()[orthant_a];
            let sub_b = tree.nodes[subdiagonal_b.into()].orthants().unwrap()[orthant_b];
            let length: Scalar = tree.nodes[a_m_e.into()].length.into();
            let offset = &Coordinate::const_from(direction) * length;
            let base_0 = coordinates[center_nodes[a_m_e.into()]].clone();
            let base_1 = coordinates[center_nodes[b_m_c.into()]].clone();
            let [n0, n1] = [&base_0 + &offset, &base_1 + &offset]
                .map(|coordinate| get_or_add(coordinate, coordinates, nodes_map, node_index));
            let a_m_e = center_nodes[a_m_e.into()];
            let a_n_f = center_nodes[a_n_f.into()];
            let b_m_c = center_nodes[b_m_c.into()];
            let b_n_d = center_nodes[b_n_d.into()];
            let subdiag_a = center_nodes[sub_a.into()];
            let subdiag_b = center_nodes[sub_b.into()];
            if flip {
                connectivity.push([n1, b_m_c, subdiag_b, b_n_d, n0, a_m_e, subdiag_a, a_n_f]);
            } else {
                connectivity.push([n0, a_m_e, subdiag_a, a_n_f, n1, b_m_c, subdiag_b, b_n_d]);
            }
        }
    }
}
