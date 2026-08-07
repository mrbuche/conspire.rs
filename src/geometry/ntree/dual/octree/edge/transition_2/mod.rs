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

/// Fills the seam between two face slabs lying in the same interface plane and abutting along an
/// edge: two clusters side by side across `seam`, each facing coarse leaves across the same
/// facet. Each slab stops at its own boundary, leaving a strip two coarse cells long uncovered.
///
/// The seam is filled from the cluster on its lower side, so each one is taken exactly once. The
/// tree-walking version instead anchored on the coarse side and picked a side per facet; which of
/// the two clusters is the anchor only mirrors the template, and the handedness factor below
/// absorbs that, so fixing the choice costs nothing.
///
/// Along the seam the four fine cells sit at positions 0..3; the middle two carry the Steiner
/// points, which this template only reads - the face slabs placed them.
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
    let mut clusters: Vec<([usize; D], usize)> = tree.pairing_vertices.iter().copied().collect();
    clusters.sort_unstable();
    for (cluster, length) in clusters {
        let center: [i64; D] = from_fn(|axis| cluster[axis] as i64);
        let (coarse, fine) = (length as i64, length as i64 / 2);
        for facet in 0..2 * D {
            let (axis, side) = (facet >> 1, facet & 1);
            let sign = if side == 1 { 1 } else { -1 };
            let interface = center[axis] + sign * coarse;
            let inside = if side == 1 {
                interface - fine
            } else {
                interface
            };
            let outside = if side == 1 {
                interface
            } else {
                interface - coarse
            };
            for seam in (0..D).filter(|&seam| seam != axis) {
                let along = D - axis - seam;
                let mut partner = cluster;
                partner[seam] += 2 * length;
                if !tree.pairing_vertices.contains(&(partner, length)) {
                    continue;
                }
                let corner_at = |across: i64, sideways: i64, lengthwise: i64| {
                    let mut corner = [0; D];
                    corner[axis] = across;
                    corner[seam] = sideways;
                    corner[along] = lengthwise;
                    corner
                };
                let column = |sideways: i64| -> [Option<usize>; 4] {
                    from_fn(|k| {
                        tree.cell_at(
                            &corner_at(inside, sideways, center[along] - coarse + k as i64 * fine),
                            fine,
                        )
                    })
                };
                let row = |sideways: i64| -> [Option<usize>; 2] {
                    from_fn(|j| {
                        tree.cell_at(
                            &corner_at(
                                outside,
                                sideways,
                                center[along] - coarse + j as i64 * coarse,
                            ),
                            coarse,
                        )
                    })
                };
                let near = column(center[seam] + coarse - fine);
                let far = column(center[seam] + coarse);
                let near_coarse = row(center[seam]);
                let far_coarse = row(center[seam] + coarse);
                if near.iter().chain(far.iter()).any(Option::is_none)
                    || near_coarse
                        .iter()
                        .chain(far_coarse.iter())
                        .any(Option::is_none)
                {
                    continue;
                }
                // Reversing the sense along the seam is what keeps every hex wound the same way
                // once the frame (outward facet, seam, edge) turns left-handed.
                let cyclic = (axis + 1) % D == seam;
                let flip = (sign == 1) != cyclic;
                let cell = |slot: Option<usize>| center_nodes[slot.unwrap()];
                let fine_at =
                    |lane: &[Option<usize>; 4], k: usize| cell(lane[if flip { 3 - k } else { k }]);
                let coarse_at =
                    |lane: &[Option<usize>; 2], j: usize| cell(lane[if flip { 1 - j } else { j }]);
                let offset = &facet_direction(facet) * (fine as Scalar);
                let steiner = |node| {
                    let coordinate = &coordinates[node] + &offset;
                    nodes_map
                        .get(&from_fn::<usize, D, _>(|i| (2.0 * coordinate[i]) as usize))
                        .copied()
                };
                let (face_a, face_b) = (fine_at(&near, 1), fine_at(&near, 2));
                let (face_c, face_d) = (fine_at(&near, 0), fine_at(&near, 3));
                let (diag_a, diag_b) = (fine_at(&far, 1), fine_at(&far, 2));
                let (diag_c, diag_d) = (fine_at(&far, 0), fine_at(&far, 3));
                let (cell_a, cell_b) = (coarse_at(&near_coarse, 1), coarse_at(&near_coarse, 0));
                let (adjacent_a, adjacent_b) =
                    (coarse_at(&far_coarse, 1), coarse_at(&far_coarse, 0));
                if let Some(node_1) = steiner(face_a)
                    && let Some(node_2) = steiner(face_b)
                    && let Some(node_3) = steiner(diag_a)
                    && let Some(node_4) = steiner(diag_b)
                {
                    connectivity.push([
                        cell_a, cell_b, node_1, node_2, adjacent_a, adjacent_b, node_3, node_4,
                    ]);
                    connectivity.push([
                        face_a, face_b, node_2, node_1, diag_a, diag_b, node_4, node_3,
                    ]);
                    connectivity.push([
                        face_b, node_2, node_4, diag_b, face_d, cell_a, adjacent_a, diag_d,
                    ]);
                    connectivity.push([
                        face_a, diag_a, node_3, node_1, face_c, diag_c, adjacent_b, cell_b,
                    ]);
                }
            }
        }
    }
}
