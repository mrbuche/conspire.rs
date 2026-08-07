use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, N, facet_direction},
            },
            node::split::Split,
        },
    },
    math::Scalar,
};
use std::array::from_fn;

/// Fills the wedge along an edge shared by two coarse leaves stacked along it, whose neighbours
/// across both adjoining facets are refined but whose diagonal neighbours are not - the
/// complement of the wedge where the diagonal is refined too.
///
/// The two coarse leaves must be paired, which is what fixes the grouping of fine cells the
/// wedge reads; the tree-walking version asked instead that they be siblings, the same statement
/// specialized to clusters that coincide with nodes.
///
/// Along the edge the four fine cells on each side sit at positions 0..3. Every Steiner point
/// here is read, not placed: two come from the slab facing the `m` neighbour and two from the
/// slab facing its own `n` neighbour.
pub(super) fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize> + Split,
    U: Copy + Into<usize>,
{
    for index in 0..tree.nodes.len() {
        let leaf = &tree.nodes[index];
        if !leaf.is_leaf() {
            continue;
        }
        let coarse = Into::<usize>::into(leaf.length) as i64;
        let fine = coarse / 2;
        if fine == 0 {
            continue;
        }
        let origin: [i64; D] = from_fn(|axis| Into::<usize>::into(leaf.corner[axis]) as i64);
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
            // The wedge is symmetric: the pair of coarse leaves across the diagonal sees the same
            // configuration mirrored, and would draw the same hexes. Taking only the corner on
            // the positive side of the higher of the two remaining axes settles it, since the two
            // views disagree there.
            let side_second = 1;
            for side_first in 0..2 {
                {
                    let cyclic = (first_axis + 1) % D == second_axis;
                    let sign = |side: usize| if side == 1 { 1i32 } else { -1 };
                    let handed =
                        sign(side_first) * sign(side_second) * if cyclic { 1 } else { -1 } == 1;
                    let (m, n, side_m, side_n) = if handed {
                        (first_axis, second_axis, side_first, side_second)
                    } else {
                        (second_axis, first_axis, side_second, side_first)
                    };
                    let corner_at = |offset_m: i64, offset_n: i64, offset_t: i64| {
                        let mut corner = origin;
                        corner[m] += offset_m;
                        corner[n] += offset_n;
                        corner[along] += offset_t;
                        corner
                    };
                    let (near_m, far_m) = if side_m == 1 {
                        (coarse - fine, coarse)
                    } else {
                        (0, -fine)
                    };
                    let (near_n, far_n) = if side_n == 1 {
                        (coarse - fine, coarse)
                    } else {
                        (0, -fine)
                    };
                    let (out_m, out_n) = (
                        if side_m == 1 { coarse } else { -coarse },
                        if side_n == 1 { coarse } else { -coarse },
                    );
                    // The two coarse leaves have to be paired, but the pairing only ever records
                    // clusters of refined cells, so it cannot say so directly. It says it about
                    // the refined columns to either side instead: the m-neighbours of the two
                    // leaves share a cluster, as do the n-neighbours, which is the same statement
                    // one cell over and is what fixes the grouping of fine cells read below.
                    if !tree.shares_cluster(&corner_at(out_m, 0, 0), coarse, along)
                        || !tree.shares_cluster(&corner_at(0, out_n, 0), coarse, along)
                    {
                        continue;
                    }
                    let side_m_cells: [Option<usize>; 4] =
                        from_fn(|k| tree.cell_at(&corner_at(far_m, near_n, k as i64 * fine), fine));
                    let side_n_cells: [Option<usize>; 4] =
                        from_fn(|k| tree.cell_at(&corner_at(near_m, far_n, k as i64 * fine), fine));
                    let diagonal: [Option<usize>; 2] = from_fn(|j| {
                        tree.cell_at(&corner_at(out_m, out_n, j as i64 * coarse), coarse)
                    });
                    if side_m_cells.iter().any(Option::is_none) {
                        continue;
                    }
                    if side_n_cells.iter().any(Option::is_none) {
                        continue;
                    }
                    if diagonal.iter().any(Option::is_none) {
                        continue;
                    }
                    let cell = |slot: Option<usize>| center_nodes[slot.unwrap()];
                    let (edge_m, edge_n): ([usize; 4], [usize; 4]) = (
                        from_fn(|k| cell(side_m_cells[k])),
                        from_fn(|k| cell(side_n_cells[k])),
                    );
                    let (diag_a, diag_b) = (cell(diagonal[0]), cell(diagonal[1]));
                    let (cell_a, cell_b) = (center_nodes[index], center_nodes[second]);
                    let step = fine as Scalar;
                    let offset_m = &facet_direction(2 * m + side_m) * step;
                    let offset_n = &facet_direction(2 * n + side_n) * step;
                    let find = |coordinate: crate::geometry::Coordinate<D>| -> Option<usize> {
                        nodes_map
                            .get(&from_fn::<usize, D, _>(|i| (2.0 * coordinate[i]) as usize))
                            .copied()
                    };
                    if let Some(node_1) = find(&coordinates[edge_m[1]] - &offset_m)
                        && let Some(node_2) = find(&coordinates[edge_m[2]] - &offset_m)
                        && let Some(node_3) = find(&coordinates[edge_m[1]] + &offset_n)
                        && let Some(node_4) = find(&coordinates[edge_m[2]] + &offset_n)
                    {
                        connectivity.push([
                            edge_m[1], node_1, node_2, edge_m[2], node_3, edge_n[1], edge_n[2],
                            node_4,
                        ]);
                        connectivity.push([
                            edge_m[2], node_2, cell_b, edge_m[3], node_4, edge_n[2], edge_n[3],
                            diag_b,
                        ]);
                        connectivity.push([
                            edge_m[0], cell_a, node_1, edge_m[1], diag_a, edge_n[0], edge_n[1],
                            node_3,
                        ]);
                    }
                }
            }
        }
    }
}
