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
                        // A half is one end of the wedge along the edge: two fine cells and one
                        // coarse cell from each of the three outside groups. Truncation can only
                        // ever split the wedge this way, since the m and n directions contribute
                        // nothing but cells hugging the edge, so a cluster truncated there loses
                        // the edge entirely rather than half of it.
                        //
                        // Take a half only when whole, and the wedge only when every absent half
                        // is provably off-domain. Gating per cell instead does not work: the
                        // first hex below reads only the two middle fine cells, so it would
                        // survive a wedge whose whole outside lies beyond the domain and draw a
                        // spurious element there.
                        let half = |j: usize| {
                            [
                                fine_cells[2 * j],
                                fine_cells[2 * j + 1],
                                face_m[j],
                                face_n[j],
                                diagonal[j],
                            ]
                        };
                        let whole = |j: usize| half(j).iter().all(Option::is_some);
                        let truncated = |j: usize| {
                            let t = j as i64 * coarse;
                            tree.off_domain(&corner_at(inner_m, inner_n, t), fine)
                                && tree.off_domain(&corner_at(inner_m, inner_n, t + fine), fine)
                                && tree.off_domain(&corner_at(outer_m, near_n, t), coarse)
                                && tree.off_domain(&corner_at(near_m, outer_n, t), coarse)
                                && tree.off_domain(&corner_at(outer_m, outer_n, t), coarse)
                        };
                        if !(0..2).any(whole) || (0..2).any(|j| !whole(j) && !truncated(j)) {
                            continue;
                        }
                        let cell = |index: Option<usize>| index.map(|index| center_nodes[index]);
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
                        let mut ring = |base: Option<usize>| {
                            base.map(|base| {
                                let base = coordinates[base].clone();
                                [
                                    &base + &offset_m,
                                    &base + &offset_m + &offset_n,
                                    &base + &offset_n,
                                ]
                                .map(|coordinate| {
                                    get_or_add(coordinate, coordinates, nodes_map, node_index)
                                })
                            })
                        };
                        let (ring_a, ring_b) = (ring(mid_a), ring(mid_b));
                        let point = |ring: Option<[usize; 3]>, k: usize| ring.map(|ring| ring[k]);
                        let (n0, n1, n2) = (point(ring_a, 0), point(ring_a, 1), point(ring_a, 2));
                        let (n3, n4, n5) = (point(ring_b, 0), point(ring_b, 1), point(ring_b, 2));
                        for hex in [
                            [mid_a, n0, n1, n2, mid_b, n3, n4, n5],
                            [n0, face_m_a, diagonal_a, n1, n3, face_m_b, diagonal_b, n4],
                            [n1, diagonal_a, face_n_a, n2, n4, diagonal_b, face_n_b, n5],
                            [mid_a, n2, n1, n0, outer_a, face_n_a, diagonal_a, face_m_a],
                            [mid_b, n3, n4, n5, outer_b, face_m_b, diagonal_b, face_n_b],
                        ] {
                            if hex.iter().all(Option::is_some) {
                                connectivity.push(from_fn(|k| hex[k].unwrap()))
                            }
                        }
                    }
                }
            }
        }
    }
}
