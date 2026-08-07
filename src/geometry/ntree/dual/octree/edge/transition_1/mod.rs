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

/// Fills the wedge along a cluster edge where the cluster's four fine cells meet two coarse
/// leaves across each adjoining facet and two more on the diagonal.
///
/// The edge runs along axis `t`; `m` and `n` are the two facets it separates, ordered so that
/// (outward-m, outward-n, +t) is right-handed, which is what keeps every hex below wound the
/// same way. Along `t` the four fine cells sit at positions 0..3: the middle two carry the
/// Steiner rings, the outer two join the coarse centres directly.
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
    let mut clusters: Vec<([usize; D], usize)> = tree.pairing_vertices.iter().copied().collect();
    clusters.sort_unstable();
    for (cluster, length) in clusters {
        let center: [i64; D] = from_fn(|axis| cluster[axis] as i64);
        let (coarse, fine) = (length as i64, length as i64 / 2);
        for first in 0..D {
            for second in first + 1..D {
                let along = D - first - second;
                for side_first in 0..2 {
                    for side_second in 0..2 {
                        // Exactly one of the two orderings is right-handed, since swapping the
                        // pair flips the permutation sign and leaves the two side signs alone.
                        let cyclic = (first + 1) % D == second;
                        let sign = |side: usize| if side == 1 { 1i32 } else { -1 };
                        let handed =
                            sign(side_first) * sign(side_second) * if cyclic { 1 } else { -1 } == 1;
                        let (m, n, side_m, side_n) = if handed {
                            (first, second, side_first, side_second)
                        } else {
                            (second, first, side_second, side_first)
                        };
                        let corner_at = |offset_m: i64, offset_n: i64, offset_t: i64| {
                            let mut corner = [0; D];
                            corner[m] = center[m] + offset_m;
                            corner[n] = center[n] + offset_n;
                            corner[along] = center[along] - coarse + offset_t;
                            corner
                        };
                        let (inner_m, inner_n) = (
                            if side_m == 1 { coarse - fine } else { -coarse },
                            if side_n == 1 { coarse - fine } else { -coarse },
                        );
                        let (outer_m, outer_n) = (
                            if side_m == 1 { coarse } else { -2 * coarse },
                            if side_n == 1 { coarse } else { -2 * coarse },
                        );
                        let (near_m, near_n) = (
                            if side_m == 1 { 0 } else { -coarse },
                            if side_n == 1 { 0 } else { -coarse },
                        );
                        let fine_cells: [Option<usize>; 4] = from_fn(|k| {
                            tree.cell_at(&corner_at(inner_m, inner_n, k as i64 * fine), fine)
                        });
                        let face_m: [Option<usize>; 2] = from_fn(|j| {
                            tree.cell_at(&corner_at(outer_m, near_n, j as i64 * coarse), coarse)
                        });
                        let face_n: [Option<usize>; 2] = from_fn(|j| {
                            tree.cell_at(&corner_at(near_m, outer_n, j as i64 * coarse), coarse)
                        });
                        let diagonal: [Option<usize>; 2] = from_fn(|j| {
                            tree.cell_at(&corner_at(outer_m, outer_n, j as i64 * coarse), coarse)
                        });
                        if fine_cells.iter().any(Option::is_none)
                            || face_m.iter().any(Option::is_none)
                            || face_n.iter().any(Option::is_none)
                            || diagonal.iter().any(Option::is_none)
                        {
                            continue;
                        }
                        let cell = |index: Option<usize>| center_nodes[index.unwrap()];
                        let (outer_a, mid_a, mid_b, outer_b) = (
                            cell(fine_cells[0]),
                            cell(fine_cells[1]),
                            cell(fine_cells[2]),
                            cell(fine_cells[3]),
                        );
                        let (face_m_a, face_m_b) = (cell(face_m[0]), cell(face_m[1]));
                        let (face_n_a, face_n_b) = (cell(face_n[0]), cell(face_n[1]));
                        let (diagonal_a, diagonal_b) = (cell(diagonal[0]), cell(diagonal[1]));
                        let step = fine as Scalar;
                        let offset_m = &facet_direction(2 * m + side_m) * step;
                        let offset_n = &facet_direction(2 * n + side_n) * step;
                        let base_a = coordinates[mid_a].clone();
                        let base_b = coordinates[mid_b].clone();
                        let [n0, n1, n2, n3, n4, n5] = [
                            &base_a + &offset_m,
                            &base_a + &offset_m + &offset_n,
                            &base_a + &offset_n,
                            &base_b + &offset_m,
                            &base_b + &offset_m + &offset_n,
                            &base_b + &offset_n,
                        ]
                        .map(|coordinate| {
                            get_or_add(coordinate, coordinates, nodes_map, node_index)
                        });
                        connectivity.push([mid_a, n0, n1, n2, mid_b, n3, n4, n5]);
                        connectivity
                            .push([n0, face_m_a, diagonal_a, n1, n3, face_m_b, diagonal_b, n4]);
                        connectivity
                            .push([n1, diagonal_a, face_n_a, n2, n4, diagonal_b, face_n_b, n5]);
                        connectivity
                            .push([mid_a, n2, n1, n0, outer_a, face_n_a, diagonal_a, face_m_a]);
                        connectivity
                            .push([mid_b, n3, n4, n5, outer_b, face_m_b, diagonal_b, face_n_b]);
                    }
                }
            }
        }
    }
}
