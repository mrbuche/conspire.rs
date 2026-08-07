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

/// Four cells along the edge, absent where the domain cuts them off.
type Lane = [Option<usize>; 4];

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
        let corner: [i64; D] = from_fn(|axis| Into::<usize>::into(first.corner[axis]) as i64);
        // Either coarse leaf of the pair may lie outside the domain, leaving half a wedge to
        // draw, so this leaf is tried as each end in turn. Only the lower role was ever
        // considered before, which silently dropped every wedge whose lower leaf was off-domain.
        for (along, lower) in (0..D).flat_map(|along| [(along, true), (along, false)]) {
            let mut origin = corner;
            if !lower {
                origin[along] -= coarse;
            }
            let mut partner = origin;
            partner[along] += coarse;
            let (first_cell, second) = if lower {
                (Some(index), tree.cell_at(&partner, coarse))
            } else {
                (tree.cell_at(&origin, coarse), Some(index))
            };
            if first_cell.is_some() != lower {
                continue;
            }
            let absent = if lower { partner } else { origin };
            if !tree.off_domain(&absent, coarse) && (first_cell.is_none() || second.is_none()) {
                continue;
            }
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
                    // A half is one end of the wedge along the edge: one coarse centre and two
                    // fine cells from each of the three refined columns. Take a half only when
                    // whole, and the wedge only when every absent half is provably off-domain.
                    let coarse_cells = [first_cell, second];
                    let half = |j: usize| {
                        [
                            side_m_cells[2 * j],
                            side_m_cells[2 * j + 1],
                            side_n_cells[2 * j],
                            side_n_cells[2 * j + 1],
                            diagonal[2 * j],
                            diagonal[2 * j + 1],
                            coarse_cells[j],
                        ]
                    };
                    let whole = |j: usize| half(j).iter().all(Option::is_some);
                    let truncated = |j: usize| {
                        (0..2).all(|k| {
                            let slice = 2 * j + k;
                            tree.off_domain(&corner_at(far_m, near_n, slice), fine)
                                && tree.off_domain(&corner_at(near_m, far_n, slice), fine)
                                && tree.off_domain(&corner_at(far_m, far_n, slice), fine)
                        }) && tree.off_domain(&corner_at(0, 0, 2 * j), coarse)
                    };
                    if !(0..2).any(whole) || (0..2).any(|j| !whole(j) && !truncated(j)) {
                        continue;
                    }
                    let cell = |slot: Option<usize>| slot.map(|slot| center_nodes[slot]);
                    let lane = |cells: &[Option<usize>; 4]| -> Lane { from_fn(|k| cell(cells[k])) };
                    let (edge_m, edge_n, edge_d) =
                        (lane(&side_m_cells), lane(&side_n_cells), lane(&diagonal));
                    let offset = &facet_direction(2 * m + (1 - side_m)) * (fine as Scalar);
                    let mut inner = |slot: Option<usize>| {
                        slot.map(|slot| {
                            let coordinate = &coordinates[slot] + &offset;
                            get_or_add(coordinate, coordinates, nodes_map, node_index)
                        })
                    };
                    let (inner_a, inner_b) = (inner(edge_m[1]), inner(edge_m[2]));
                    let slices = [
                        [cell(coarse_cells[0]), edge_m[0], edge_d[0], edge_n[0]],
                        [inner_a, edge_m[1], edge_d[1], edge_n[1]],
                        [inner_b, edge_m[2], edge_d[2], edge_n[2]],
                        [cell(coarse_cells[1]), edge_m[3], edge_d[3], edge_n[3]],
                    ];
                    for pair in slices.windows(2) {
                        let hex = [
                            pair[0][0], pair[0][1], pair[0][2], pair[0][3], pair[1][0], pair[1][1],
                            pair[1][2], pair[1][3],
                        ];
                        if hex.iter().all(Option::is_some) {
                            connectivity.push(from_fn(|k| hex[k].unwrap()))
                        }
                    }
                }
            }
        }
    }
}
