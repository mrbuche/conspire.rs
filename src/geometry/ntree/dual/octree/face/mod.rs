use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, leaf_containing,
                octree::{D, L, M, N},
            },
            node::split::Split,
        },
    },
    math::{Scalar, TensorVec},
};
use std::array::from_fn;

const LL: usize = L * L;
const SCALE_1: Scalar = 0.5;

/// Grid positions, within the facet's 4x4 array of fine cells, that the template anchors its
/// interior points on (the inner 2x2) and its exterior points on (the surrounding ring).
const INTERIOR_ANCHORS: [usize; 4] = [3, 6, 12, 9];
const EXTERIOR_ANCHORS: [usize; 8] = [1, 4, 7, 13, 14, 11, 8, 2];

/// Fills the slab facing each facet of each paired cluster, where the cluster's sixteen fine cells
/// meet four coarse leaves.
///
/// Anchoring on the cluster rather than on a tree node is what lets this fire for
/// `Pairing::Generalized`, whose clusters need not coincide with nodes. A cluster may also hang off
/// the domain, leaving only half or a quarter of the facet real. Rather than author a template
/// per truncation, the full template is emitted with missing cells threaded through as `None`
/// and any hex touching one dropped: every hex pairs each of its cells with a Steiner point at
/// the same grid position, so a hex whose cells all exist always has its points too, and what
/// survives is exactly the truncated template.
pub(super) fn face_transition<T, U>(
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
    let root = &tree.nodes[0];
    let low: [i64; D] = from_fn(|axis| Into::<usize>::into(root.corner[axis]) as i64);
    let high: [i64; D] = from_fn(|axis| low[axis] + Into::<usize>::into(root.length) as i64);
    let cell_at = |corner: [i64; D], length: i64| -> Option<usize> {
        if (0..D).any(|axis| corner[axis] < low[axis] || corner[axis] + length > high[axis]) {
            return None;
        }
        let point = from_fn(|axis| corner[axis] as usize);
        let index = leaf_containing(tree, &point);
        let node = &tree.nodes[index];
        (length as usize == node.length.into()
            && (0..D).all(|axis| point[axis] == node.corner[axis].into()))
        .then_some(index)
    };
    let mut clusters: Vec<([usize; D], usize)> = tree.pairing_vertices.iter().copied().collect();
    clusters.sort_unstable();
    for (cluster, length) in clusters {
        let center: [i64; D] = from_fn(|axis| cluster[axis] as i64);
        let (coarse, fine) = (length as i64, length as i64 / 2);
        for facet in 0..M {
            let (axis, side) = (facet >> 1, facet & 1);
            let tangents: [usize; 2] = {
                let mut others = (0..D).filter(|&other| other != axis);
                [others.next().unwrap(), others.next().unwrap()]
            };
            let interface = center[axis] + if side == 1 { coarse } else { -coarse };
            let outside = if side == 1 {
                interface
            } else {
                interface - coarse
            };
            let inside = if side == 1 {
                interface - fine
            } else {
                interface
            };
            let corner_at = |along: i64, first: i64, second: i64| {
                let mut corner = [0; D];
                corner[axis] = along;
                corner[tangents[0]] = center[tangents[0]] - coarse + first;
                corner[tangents[1]] = center[tangents[1]] - coarse + second;
                corner
            };
            // A quarter is one coarse leaf together with the 2x2 fine cells behind it. It is
            // taken only when whole; a quarter that is absent for any reason other than lying
            // outside the domain means the real transition belongs to some other cluster, and
            // this facet is left alone entirely.
            let quarter_off_domain = |first: usize, second: usize| {
                let u = center[tangents[0]] - coarse + first as i64 * coarse;
                let v = center[tangents[1]] - coarse + second as i64 * coarse;
                u < low[tangents[0]]
                    || u + coarse > high[tangents[0]]
                    || v < low[tangents[1]]
                    || v + coarse > high[tangents[1]]
            };
            let mut coarse_cells: [Option<usize>; L] = [None; L];
            let mut fine_cells: [Option<usize>; LL] = [None; LL];
            let (mut usable, mut any) = (true, false);
            for quarter in 0..L {
                let (first, second) = (quarter & 1, quarter >> 1);
                let coarse_cell = cell_at(
                    corner_at(outside, first as i64 * coarse, second as i64 * coarse),
                    coarse,
                );
                let quarter_fine: [Option<usize>; L] = from_fn(|k| {
                    cell_at(
                        corner_at(
                            inside,
                            (2 * first + (k & 1)) as i64 * fine,
                            (2 * second + (k >> 1)) as i64 * fine,
                        ),
                        fine,
                    )
                });
                if coarse_cell.is_some() && quarter_fine.iter().all(Option::is_some) {
                    any = true;
                    coarse_cells[quarter] = coarse_cell;
                    // Within a quarter the fine cells land contiguously, since the facet is
                    // indexed orthant-major and the quarter is the orthant.
                    fine_cells[L * quarter..L * (quarter + 1)].copy_from_slice(&quarter_fine);
                } else if !quarter_off_domain(first, second) {
                    usable = false;
                }
            }
            if usable && any {
                template(
                    coarse_cells,
                    center_nodes,
                    nodes_map,
                    facet ^ 1,
                    fine_cells,
                    tree,
                    connectivity,
                    coordinates,
                    node_index,
                )
            }
        }
    }
}

fn push(connectivity: &mut Vec<[usize; N]>, hex: [Option<usize>; N]) {
    if hex.iter().all(Option::is_some) {
        connectivity.push(from_fn(|k| hex[k].unwrap()))
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    leaves: [Option<usize>; L],
    center_nodes: &[usize],
    nodes_map: &mut NodeMap<D>,
    facet: usize,
    neighbors: [Option<usize>; LL],
    tree: &Octree<T, U>,
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    node_index: &mut usize,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let Some(&present) = neighbors.iter().flatten().next() else {
        return;
    };
    let leaves_center_nodes: [Option<usize>; L] =
        from_fn(|i| leaves[i].map(|cell| center_nodes[cell]));
    let neighbors_center: [Option<usize>; LL] =
        from_fn(|k| neighbors[k].map(|cell| center_nodes[cell]));
    let (scale_1, scale_2) = translations(facet, tree.nodes[present].length.into());
    let interior_nodes: [Option<usize>; 4] = from_fn(|j| {
        neighbors_center[INTERIOR_ANCHORS[j]].map(|adjacent| {
            let coordinate = &coordinates[adjacent] + &scale_1;
            let node = *node_index;
            coordinates.push(coordinate);
            *node_index += 1;
            node
        })
    });
    let exterior_nodes: [Option<usize>; 8] = from_fn(|j| {
        neighbors_center[EXTERIOR_ANCHORS[j]].map(|adjacent| {
            let coordinate = &coordinates[adjacent] + &scale_2;
            let indices = from_fn(|i| (2.0 * coordinate[i]) as usize);
            if let Some(&node) = nodes_map.get(&indices) {
                node
            } else {
                let node = *node_index;
                coordinates.push(coordinate);
                nodes_map.insert(indices, node);
                *node_index += 1;
                node
            }
        })
    });
    connectivity_template(
        leaves_center_nodes,
        facet,
        neighbors_center,
        interior_nodes,
        exterior_nodes,
        connectivity,
    )
}

fn connectivity_template(
    leaves_center_nodes: [Option<usize>; L],
    facet: usize,
    neighbors_center: [Option<usize>; LL],
    interior_nodes: [Option<usize>; 4],
    exterior_nodes: [Option<usize>; 8],
    connectivity: &mut Vec<[usize; N]>,
) {
    match facet {
        0 | 3 | 4 => {
            push(
                connectivity,
                [
                    neighbors_center[3],
                    neighbors_center[6],
                    neighbors_center[12],
                    neighbors_center[9],
                    interior_nodes[0],
                    interior_nodes[1],
                    interior_nodes[2],
                    interior_nodes[3],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[1],
                    neighbors_center[4],
                    neighbors_center[6],
                    neighbors_center[3],
                    exterior_nodes[0],
                    exterior_nodes[1],
                    interior_nodes[1],
                    interior_nodes[0],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[6],
                    neighbors_center[7],
                    neighbors_center[13],
                    neighbors_center[12],
                    interior_nodes[1],
                    exterior_nodes[2],
                    exterior_nodes[3],
                    interior_nodes[2],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[9],
                    neighbors_center[12],
                    neighbors_center[14],
                    neighbors_center[11],
                    interior_nodes[3],
                    interior_nodes[2],
                    exterior_nodes[4],
                    exterior_nodes[5],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[2],
                    neighbors_center[3],
                    neighbors_center[9],
                    neighbors_center[8],
                    exterior_nodes[7],
                    interior_nodes[0],
                    interior_nodes[3],
                    exterior_nodes[6],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[0],
                    neighbors_center[1],
                    neighbors_center[3],
                    neighbors_center[2],
                    leaves_center_nodes[0],
                    exterior_nodes[0],
                    interior_nodes[0],
                    exterior_nodes[7],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[4],
                    neighbors_center[5],
                    neighbors_center[7],
                    neighbors_center[6],
                    exterior_nodes[1],
                    leaves_center_nodes[1],
                    exterior_nodes[2],
                    interior_nodes[1],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[12],
                    neighbors_center[13],
                    neighbors_center[15],
                    neighbors_center[14],
                    interior_nodes[2],
                    exterior_nodes[3],
                    leaves_center_nodes[3],
                    exterior_nodes[4],
                ],
            );
            push(
                connectivity,
                [
                    neighbors_center[8],
                    neighbors_center[9],
                    neighbors_center[11],
                    neighbors_center[10],
                    exterior_nodes[6],
                    interior_nodes[3],
                    exterior_nodes[5],
                    leaves_center_nodes[2],
                ],
            );
            push(
                connectivity,
                [
                    interior_nodes[0],
                    interior_nodes[1],
                    interior_nodes[2],
                    interior_nodes[3],
                    exterior_nodes[0],
                    exterior_nodes[1],
                    exterior_nodes[4],
                    exterior_nodes[5],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[0],
                    exterior_nodes[1],
                    exterior_nodes[4],
                    exterior_nodes[5],
                    leaves_center_nodes[0],
                    leaves_center_nodes[1],
                    leaves_center_nodes[3],
                    leaves_center_nodes[2],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[2],
                    exterior_nodes[3],
                    interior_nodes[2],
                    interior_nodes[1],
                    leaves_center_nodes[1],
                    leaves_center_nodes[3],
                    exterior_nodes[4],
                    exterior_nodes[1],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[6],
                    exterior_nodes[7],
                    interior_nodes[0],
                    interior_nodes[3],
                    leaves_center_nodes[2],
                    leaves_center_nodes[0],
                    exterior_nodes[0],
                    exterior_nodes[5],
                ],
            );
        }
        1 | 2 | 5 => {
            push(
                connectivity,
                [
                    interior_nodes[0],
                    interior_nodes[1],
                    interior_nodes[2],
                    interior_nodes[3],
                    neighbors_center[3],
                    neighbors_center[6],
                    neighbors_center[12],
                    neighbors_center[9],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[0],
                    exterior_nodes[1],
                    interior_nodes[1],
                    interior_nodes[0],
                    neighbors_center[1],
                    neighbors_center[4],
                    neighbors_center[6],
                    neighbors_center[3],
                ],
            );
            push(
                connectivity,
                [
                    interior_nodes[1],
                    exterior_nodes[2],
                    exterior_nodes[3],
                    interior_nodes[2],
                    neighbors_center[6],
                    neighbors_center[7],
                    neighbors_center[13],
                    neighbors_center[12],
                ],
            );
            push(
                connectivity,
                [
                    interior_nodes[3],
                    interior_nodes[2],
                    exterior_nodes[4],
                    exterior_nodes[5],
                    neighbors_center[9],
                    neighbors_center[12],
                    neighbors_center[14],
                    neighbors_center[11],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[7],
                    interior_nodes[0],
                    interior_nodes[3],
                    exterior_nodes[6],
                    neighbors_center[2],
                    neighbors_center[3],
                    neighbors_center[9],
                    neighbors_center[8],
                ],
            );
            push(
                connectivity,
                [
                    leaves_center_nodes[0],
                    exterior_nodes[0],
                    interior_nodes[0],
                    exterior_nodes[7],
                    neighbors_center[0],
                    neighbors_center[1],
                    neighbors_center[3],
                    neighbors_center[2],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[1],
                    leaves_center_nodes[1],
                    exterior_nodes[2],
                    interior_nodes[1],
                    neighbors_center[4],
                    neighbors_center[5],
                    neighbors_center[7],
                    neighbors_center[6],
                ],
            );
            push(
                connectivity,
                [
                    interior_nodes[2],
                    exterior_nodes[3],
                    leaves_center_nodes[3],
                    exterior_nodes[4],
                    neighbors_center[12],
                    neighbors_center[13],
                    neighbors_center[15],
                    neighbors_center[14],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[6],
                    interior_nodes[3],
                    exterior_nodes[5],
                    leaves_center_nodes[2],
                    neighbors_center[8],
                    neighbors_center[9],
                    neighbors_center[11],
                    neighbors_center[10],
                ],
            );
            push(
                connectivity,
                [
                    exterior_nodes[0],
                    exterior_nodes[1],
                    exterior_nodes[4],
                    exterior_nodes[5],
                    interior_nodes[0],
                    interior_nodes[1],
                    interior_nodes[2],
                    interior_nodes[3],
                ],
            );
            push(
                connectivity,
                [
                    leaves_center_nodes[0],
                    leaves_center_nodes[1],
                    leaves_center_nodes[3],
                    leaves_center_nodes[2],
                    exterior_nodes[0],
                    exterior_nodes[1],
                    exterior_nodes[4],
                    exterior_nodes[5],
                ],
            );
            push(
                connectivity,
                [
                    leaves_center_nodes[1],
                    leaves_center_nodes[3],
                    exterior_nodes[4],
                    exterior_nodes[1],
                    exterior_nodes[2],
                    exterior_nodes[3],
                    interior_nodes[2],
                    interior_nodes[1],
                ],
            );
            push(
                connectivity,
                [
                    leaves_center_nodes[2],
                    leaves_center_nodes[0],
                    exterior_nodes[0],
                    exterior_nodes[5],
                    exterior_nodes[6],
                    exterior_nodes[7],
                    interior_nodes[0],
                    interior_nodes[3],
                ],
            );
        }
        _ => unreachable!(),
    }
}

fn translations(facet: usize, length: Scalar) -> (Coordinate<D>, Coordinate<D>) {
    let (axis, side) = (facet >> 1, facet & 1);
    let sign = if side == 1 { -1.0 } else { 1.0 };
    let mut near = [0.0; D];
    let mut far = [0.0; D];
    near[axis] = sign * SCALE_1 * length;
    far[axis] = sign * length;
    (near.into(), far.into())
}
