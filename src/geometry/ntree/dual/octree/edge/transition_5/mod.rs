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
            );
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
            // Inner corner toward -facet_m, -facet_n; a_bit/b_bit select the
            // subcell side along the edge axis.
            let axis_m = facet_m >> 1;
            let axis_n = facet_n >> 1;
            let axis_t = 3 - axis_m - axis_n;

            let corner = ((1 - (facet_m & 1)) << axis_m)
                | ((1 - (facet_n & 1)) << axis_n);

            let a_bit = ((subcell_a >> axis_t) & 1) << axis_t;
            let b_bit = ((subcell_b >> axis_t) & 1) << axis_t;

            let dir_m = facet_direction(facet_m);
            let dir_n = facet_direction(facet_n);
            let dir_t = facet_direction(2 * axis_t + 1);

            // Center hex: transition_3's middle hex, but subdiagonals are
            // refined one level, so use selected children.
            let sub_a = tree.nodes[subdiagonal_a.into()]
                .orthants()
                .unwrap()[corner | b_bit];

            let sub_b = tree.nodes[subdiagonal_b.into()]
                .orthants()
                .unwrap()[corner | a_bit];

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
                    n1,
                    center_b_m_c,
                    subdiag_b,
                    center_b_n_d,
                    n0,
                    center_a_m_e,
                    subdiag_a,
                    center_a_n_f,
                ]);
            } else {
                connectivity.push([
                    n0,
                    center_a_m_e,
                    subdiag_a,
                    center_a_n_f,
                    n1,
                    center_b_m_c,
                    subdiag_b,
                    center_b_n_d,
                ]);
            }

            // -----------------------------------------------------------------
            // A side
            // -----------------------------------------------------------------
            //
            // Chain, non-flipped:
            //
            //     A-side cube high face
            //         -> A bridge
            //             -> center A face
            //                 -> center hex
            //
            // The A-side center face is:
            //
            //     [n0, center_a_m_e, subdiag_a, center_a_n_f]
            //
            // The A-side cube face adjacent to that is the high ring of
            // cube_hi_a:
            //
            //     [a_hi, b_hi, d_hi, c_hi]
            //
            let cube_lo_a = tree.nodes[diagonal_a.into()]
                .orthants()
                .unwrap()[corner | b_bit];

            let cube_hi_a = tree.nodes[subdiagonal_a.into()]
                .orthants()
                .unwrap()[corner | a_bit];

            let [a_hi, b_hi, d_hi, c_hi] = side_lo_ring(
                cube_hi_a,
                &dir_m,
                &dir_n,
                tree,
                center_nodes,
                coordinates,
                node_index,
                nodes_map,
            );

            if flip {
                connectivity.push([
                    n0,
                    center_a_m_e,
                    subdiag_a,
                    center_a_n_f,
                    a_hi,
                    b_hi,
                    d_hi,
                    c_hi,
                ]);
            } else {
                connectivity.push([
                    a_hi,
                    b_hi,
                    d_hi,
                    c_hi,
                    n0,
                    center_a_m_e,
                    subdiag_a,
                    center_a_n_f,
                ]);
            }

            push_side(
                cube_lo_a,
                cube_hi_a,
                center_nodes[node_a.into()],
                center_nodes[a_m_c.into()],
                center_nodes[a_m_e.into()],
                center_nodes[a_n_d.into()],
                center_nodes[a_n_f.into()],
                &dir_m,
                &dir_n,
                &dir_t,
                flip,
                tree,
                center_nodes,
                coordinates,
                connectivity,
                node_index,
                nodes_map,
            );

            // -----------------------------------------------------------------
            // B side
            // -----------------------------------------------------------------
            //
            // Chain, non-flipped:
            //
            //     center hex
            //         -> center B face
            //             -> B bridge
            //                 -> B-side cube low face
            //
            // The B-side center face is:
            //
            //     [n1, center_b_m_c, subdiag_b, center_b_n_d]
            //
            // The B-side cube face adjacent to that is the low ring of
            // cube_lo_b:
            //
            //     [a_lo, b_lo, d_lo, c_lo]
            //
            let cube_lo_b = tree.nodes[subdiagonal_b.into()]
                .orthants()
                .unwrap()[corner | b_bit];

            let cube_hi_b = tree.nodes[diagonal_b.into()]
                .orthants()
                .unwrap()[corner | a_bit];

            let [a_lo, b_lo, d_lo, c_lo] = side_lo_ring(
                cube_lo_b,
                &dir_m,
                &dir_n,
                tree,
                center_nodes,
                coordinates,
                node_index,
                nodes_map,
            );

            if flip {
                connectivity.push([
                    a_lo,
                    b_lo,
                    d_lo,
                    c_lo,
                    n1,
                    center_b_m_c,
                    subdiag_b,
                    center_b_n_d,
                ]);
            } else {
                connectivity.push([
                    n1,
                    center_b_m_c,
                    subdiag_b,
                    center_b_n_d,
                    a_lo,
                    b_lo,
                    d_lo,
                    c_lo,
                ]);
            }

            push_side(
                cube_lo_b,
                cube_hi_b,
                center_nodes[node_b.into()],
                center_nodes[b_m_c.into()],
                center_nodes[b_m_e.into()],
                center_nodes[b_n_d.into()],
                center_nodes[b_n_f.into()],
                &dir_m,
                &dir_n,
                &dir_t,
                flip,
                tree,
                center_nodes,
                coordinates,
                connectivity,
                node_index,
                nodes_map,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn side_lo_ring<T, U>(
    d_cell: U,
    dir_m: &Coordinate<D>,
    dir_n: &Coordinate<D>,
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) -> [usize; 4]
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let length: Scalar = tree.nodes[d_cell.into()].length.into();

    let center = coordinates[center_nodes[d_cell.into()]].clone();

    let off_m = dir_m * length;
    let off_n = dir_n * length;

    let a_c = &(&center - &off_m) - &off_n;
    let b_c = &center - &off_n;
    let c_c = &center - &off_m;

    let a = get_or_add(a_c, coordinates, nodes_map, node_index);
    let b = get_or_add(b_c, coordinates, nodes_map, node_index);
    let d = center_nodes[d_cell.into()];
    let c = get_or_add(c_c, coordinates, nodes_map, node_index);

    [a, b, d, c]
}

// One side of the fill: the perfect cube around two stacked D cells plus its
// B-ward and C-ward laterals out to the coarse center. The cube's six ring
// corners dedupe via get_or_add; the four toward B/C reuse the face-transition
// nodes, and the two toward A plus the single S node are created.
#[allow(clippy::too_many_arguments)]
fn push_side<T, U>(
    d_lo_cell: U,
    d_hi_cell: U,
    node_center: usize,
    m_lo: usize,
    m_hi: usize,
    n_lo: usize,
    n_hi: usize,
    dir_m: &Coordinate<D>,
    dir_n: &Coordinate<D>,
    dir_t: &Coordinate<D>,
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
    let half = 0.5 * length;

    let a_lo_c = &(&lo - &off_m) - &off_n;
    let a_hi_c = &(&hi - &off_m) - &off_n;

    let s_c = &(&(&a_lo_c - &(dir_m * half)) - &(dir_n * half)) - &(dir_t * half);

    let b_lo_c = &lo - &off_n;
    let c_lo_c = &lo - &off_m;
    let b_hi_c = &hi - &off_n;
    let c_hi_c = &hi - &off_m;

    let a_lo = get_or_add(a_lo_c, coordinates, nodes_map, node_index);
    let b_lo = get_or_add(b_lo_c, coordinates, nodes_map, node_index);
    let c_lo = get_or_add(c_lo_c, coordinates, nodes_map, node_index);

    let a_hi = get_or_add(a_hi_c, coordinates, nodes_map, node_index);
    let b_hi = get_or_add(b_hi_c, coordinates, nodes_map, node_index);
    let c_hi = get_or_add(c_hi_c, coordinates, nodes_map, node_index);

    let s = get_or_add(s_c, coordinates, nodes_map, node_index);

    let d_lo = center_nodes[d_lo_cell.into()];
    let d_hi = center_nodes[d_hi_cell.into()];

    if flip {
        connectivity.push([
            a_hi,
            b_hi,
            d_hi,
            c_hi,
            a_lo,
            b_lo,
            d_lo,
            c_lo,
        ]);

        connectivity.push([
            s,
            a_lo,
            b_lo,
            m_lo,
            node_center,
            a_hi,
            b_hi,
            m_hi,
        ]);

        connectivity.push([
            c_lo,
            a_lo,
            s,
            n_lo,
            c_hi,
            a_hi,
            node_center,
            n_hi,
        ]);
    } else {
        connectivity.push([
            a_lo,
            b_lo,
            d_lo,
            c_lo,
            a_hi,
            b_hi,
            d_hi,
            c_hi,
        ]);

        connectivity.push([
            s,
            m_lo,
            b_lo,
            a_lo,
            node_center,
            m_hi,
            b_hi,
            a_hi,
        ]);

        connectivity.push([
            c_lo,
            n_lo,
            s,
            a_lo,
            c_hi,
            n_hi,
            node_center,
            a_hi,
        ]);
    }
}