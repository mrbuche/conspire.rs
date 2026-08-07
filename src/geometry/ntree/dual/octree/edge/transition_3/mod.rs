use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, N, facet_direction},
            },
            node::split::Split,
        },
    },
    math::Scalar,
};
use std::array::from_fn;

/// Fills the wedge along an edge shared by two coarse leaves stacked along the edge, whose
/// neighbours across both adjoining facets, and across the diagonal, are all refined.
///
/// Read as four slices stacked along the edge: each slice is a quad of (coarse centre or Steiner
/// point, m-side fine cell, diagonal fine cell, n-side fine cell), and consecutive slices bound
/// one hex. The outer slices sit on the two coarse centres, the inner two on Steiner points
/// pushed a fine length back into those coarse cells so they straddle the shared face.
///
/// The three refined columns around the coarse pair belong to as many as three different
/// clusters, so unlike the neighbouring wedge this cannot be anchored on one of them. What ties
/// the configuration together is that each column's two coarse cells are themselves paired: that
/// is what fixes the grouping of four fine cells the slices rely on. Requiring it directly is
/// what generalizes the tree-walking version's demand that the two coarse leaves be siblings -
/// siblinghood is the same statement, specialized to clusters that coincide with nodes.
pub(super) fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize> + Split,
    U: Copy + Into<usize>,
{
    for index in 0..tree.nodes.len() {
        let first = &tree.nodes[index];
        if !first.is_leaf() {
            continue;
        }
        let coarse = Into::<usize>::into(first.length) as i64;
        let fine = coarse / 2;
        if fine == 0 {
            continue;
        }
        let origin: [i64; D] = from_fn(|axis| Into::<usize>::into(first.corner[axis]) as i64);
        for along in 0..D {
            let mut partner = origin;
            partner[along] += coarse;
            let Some(second) = tree.cell_at(&partner, coarse) else {
                continue;
            };
            let (first_axis, second_axis) = {
                let mut others = (0..D).filter(|&other| other != along);
                (others.next().unwrap(), others.next().unwrap())
            };
            for side_first in 0..2 {
                for side_second in 0..2 {
                    // As in the neighbouring wedge, exactly one ordering of the two facets is
                    // right-handed, so pick it rather than carrying a flip flag.
                    let cyclic = (first_axis + 1) % D == second_axis;
                    let sign = |side: usize| if side == 1 { 1i32 } else { -1 };
                    let handed =
                        sign(side_first) * sign(side_second) * if cyclic { 1 } else { -1 } == 1;
                    let (m, n, side_m, side_n) = if handed {
                        (first_axis, second_axis, side_first, side_second)
                    } else {
                        (second_axis, first_axis, side_second, side_first)
                    };
                    let (far_m, near_m) = if side_m == 1 {
                        (coarse, fine)
                    } else {
                        (-fine, 0)
                    };
                    let (far_n, near_n) = if side_n == 1 {
                        (coarse, fine)
                    } else {
                        (-fine, 0)
                    };
                    let (step_m, step_n) = (
                        if side_m == 1 { coarse } else { -coarse },
                        if side_n == 1 { coarse } else { -coarse },
                    );
                    let column = |offset_m: i64, offset_n: i64| {
                        let mut corner = origin;
                        corner[m] += offset_m;
                        corner[n] += offset_n;
                        corner
                    };
                    if !tree.shares_cluster(&column(step_m, 0), coarse, along)
                        || !tree.shares_cluster(&column(0, step_n), coarse, along)
                        || !tree.shares_cluster(&column(step_m, step_n), coarse, along)
                    {
                        continue;
                    }
                    let corner_at = |offset_m: i64, offset_n: i64, slice: usize| {
                        let mut corner = origin;
                        corner[m] += offset_m;
                        corner[n] += offset_n;
                        corner[along] += slice as i64 * fine;
                        corner
                    };
                    let side_m_cells: [Option<usize>; 4] =
                        from_fn(|k| tree.cell_at(&corner_at(far_m, near_n, k), fine));
                    let side_n_cells: [Option<usize>; 4] =
                        from_fn(|k| tree.cell_at(&corner_at(near_m, far_n, k), fine));
                    let diagonal: [Option<usize>; 4] =
                        from_fn(|k| tree.cell_at(&corner_at(far_m, far_n, k), fine));
                    if side_m_cells.iter().any(Option::is_none)
                        || side_n_cells.iter().any(Option::is_none)
                        || diagonal.iter().any(Option::is_none)
                    {
                        continue;
                    }
                    let cell = |slot: Option<usize>| center_nodes[slot.unwrap()];
                    let (edge_m, edge_n, edge_d): ([usize; 4], [usize; 4], [usize; 4]) = (
                        from_fn(|k| cell(side_m_cells[k])),
                        from_fn(|k| cell(side_n_cells[k])),
                        from_fn(|k| cell(diagonal[k])),
                    );
                    let offset = &facet_direction(2 * m + (1 - side_m)) * (fine as Scalar);
                    let [inner_a, inner_b] = [
                        &coordinates[edge_m[1]] + &offset,
                        &coordinates[edge_m[2]] + &offset,
                    ]
                    .map(|coordinate| get_or_add(coordinate, coordinates, nodes_map, node_index));
                    let slices = [
                        [center_nodes[index], edge_m[0], edge_d[0], edge_n[0]],
                        [inner_a, edge_m[1], edge_d[1], edge_n[1]],
                        [inner_b, edge_m[2], edge_d[2], edge_n[2]],
                        [center_nodes[second], edge_m[3], edge_d[3], edge_n[3]],
                    ];
                    for pair in slices.windows(2) {
                        connectivity.push([
                            pair[0][0], pair[0][1], pair[0][2], pair[0][3], pair[1][0], pair[1][1],
                            pair[1][2], pair[1][3],
                        ])
                    }
                }
            }
        }
    }
}
