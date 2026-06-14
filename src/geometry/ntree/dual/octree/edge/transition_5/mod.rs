use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, N, facet_direction},
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
            // inner corner toward -facet_m, -facet_n; a_bit/b_bit select the
            // subcell side along the edge axis
            let axis_m = facet_m >> 1;
            let axis_n = facet_n >> 1;
            let axis_t = 3 - axis_m - axis_n;
            let corner = ((1 - (facet_m & 1)) << axis_m) | ((1 - (facet_n & 1)) << axis_n);
            let a_bit = ((subcell_a >> axis_t) & 1) << axis_t;
            let b_bit = ((subcell_b >> axis_t) & 1) << axis_t;
            let dir_m = facet_direction(facet_m);
            let dir_n = facet_direction(facet_n);
            // center hex (transition_3's middle hex; subdiagonals refined one level)
            let sub_a = tree.nodes[subdiagonal_a.into()].orthants().unwrap()[corner | b_bit];
            let sub_b = tree.nodes[subdiagonal_b.into()].orthants().unwrap()[corner | a_bit];
            let length: Scalar = tree.nodes[a_m_e.into()].length.into();
            let offset = &Coordinate::const_from(direction) * length;
            let base_0 = coordinates[center_nodes[a_m_e.into()]].clone();
            let base_1 = coordinates[center_nodes[b_m_c.into()]].clone();
            let [n0, n1] = [&base_0 + &offset, &base_1 + &offset]
                .map(|coordinate| get_or_add(coordinate, coordinates, nodes_map, node_index));
            let center_a_m_e = center_nodes[a_m_e.into()];
            let center_a_n_f = center_nodes[a_n_f.into()];
            let center_b_m_c = center_nodes[b_m_c.into()];
            let center_b_n_d = center_nodes[b_n_d.into()];
            let subdiag_a = center_nodes[sub_a.into()];
            let subdiag_b = center_nodes[sub_b.into()];
            if flip {
                connectivity.push([
                    n1, center_b_m_c, subdiag_b, center_b_n_d, n0, center_a_m_e, subdiag_a,
                    center_a_n_f,
                ]);
            } else {
                connectivity.push([
                    n0, center_a_m_e, subdiag_a, center_a_n_f, n1, center_b_m_c, subdiag_b,
                    center_b_n_d,
                ]);
            }
            // side cube hexes: the perfect cube around each coarse leaf's middle
            // two D cells (the diagonal's inner children)
            let cube_lo_a = tree.nodes[diagonal_a.into()].orthants().unwrap()[corner | b_bit];
            let cube_hi_a = tree.nodes[subdiagonal_a.into()].orthants().unwrap()[corner | a_bit];
            push_cube(
                cube_lo_a, cube_hi_a, &dir_m, &dir_n, flip, tree, center_nodes, coordinates,
                connectivity, node_index, nodes_map,
            );
            let cube_lo_b = tree.nodes[subdiagonal_b.into()].orthants().unwrap()[corner | b_bit];
            let cube_hi_b = tree.nodes[diagonal_b.into()].orthants().unwrap()[corner | a_bit];
            push_cube(
                cube_lo_b, cube_hi_b, &dir_m, &dir_n, flip, tree, center_nodes, coordinates,
                connectivity, node_index, nodes_map,
            );
        }
    }
}

// the perfect cube around two stacked D cells (d_lo, d_hi): the six ring corners
// at the finest spacing are deduped via get_or_add (the four toward B/C reuse the
// face-transition nodes; only the two toward A are created)
#[allow(clippy::too_many_arguments)]
fn push_cube<T, U>(
    d_lo_cell: U,
    d_hi_cell: U,
    dir_m: &Coordinate<D>,
    dir_n: &Coordinate<D>,
    flip: bool,
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
    let length: Scalar = tree.nodes[d_lo_cell.into()].length.into();
    let lo = coordinates[center_nodes[d_lo_cell.into()]].clone();
    let hi = coordinates[center_nodes[d_hi_cell.into()]].clone();
    let off_m = dir_m * length;
    let off_n = dir_n * length;
    let [a_lo, b_lo, c_lo, a_hi, b_hi, c_hi] = [
        &(&lo - &off_m) - &off_n,
        &lo - &off_n,
        &lo - &off_m,
        &(&hi - &off_m) - &off_n,
        &hi - &off_n,
        &hi - &off_m,
    ]
    .map(|coordinate| get_or_add(coordinate, coordinates, nodes_map, node_index));
    let d_lo = center_nodes[d_lo_cell.into()];
    let d_hi = center_nodes[d_hi_cell.into()];
    if flip {
        connectivity.push([a_hi, b_hi, d_hi, c_hi, a_lo, b_lo, d_lo, c_lo]);
    } else {
        connectivity.push([a_lo, b_lo, d_lo, c_lo, a_hi, b_hi, d_hi, c_hi]);
    }
}
