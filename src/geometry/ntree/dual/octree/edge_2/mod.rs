use super::edge_1::facet_direction;
use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, L, M, N},
            },
            node::Node,
        },
    },
    math::Scalar,
};
use std::array::from_fn;

const LL: usize = L * L;

// Indexed by facet; each facet has two rows of thirteen:
// [adjacent_facet, cell_a, cell_b, adjacent_cell_a, adjacent_cell_b,
//  face_a, face_b, face_c, face_d, diag_face_a, diag_face_b, diag_face_c, diag_face_d].
// `cell_*`/`adjacent_cell_*` index the eight leaf children of the (coarse) cell and
// its `adjacent_facet` neighbor; `face_*`/`diag_face_*` index the sixteen sub-subcells
// on the shared face of the (fine) `facet` neighbor and the diagonal neighbor.
const EDGES: [[[usize; 13]; 2]; 6] = [
    [
        [3, 6, 2, 4, 0, 7, 13, 5, 15, 2, 8, 0, 10],
        [5, 4, 6, 0, 2, 14, 11, 15, 10, 4, 1, 5, 0],
    ],
    [
        [2, 5, 1, 7, 3, 2, 8, 0, 10, 7, 13, 5, 15],
        [4, 1, 3, 5, 7, 4, 1, 5, 0, 14, 11, 15, 10],
    ],
    [
        [1, 1, 5, 0, 4, 13, 7, 15, 5, 8, 2, 10, 0],
        [4, 0, 1, 4, 5, 4, 1, 5, 0, 14, 11, 15, 10],
    ],
    [
        [1, 7, 3, 6, 2, 7, 13, 5, 15, 2, 8, 0, 10],
        [5, 6, 7, 2, 3, 14, 11, 15, 10, 4, 1, 5, 0],
    ],
    [
        [1, 3, 1, 2, 0, 7, 13, 5, 15, 2, 8, 0, 10],
        [3, 2, 3, 0, 1, 14, 11, 15, 10, 4, 1, 5, 0],
    ],
    [
        [2, 4, 5, 6, 7, 4, 1, 5, 0, 14, 11, 15, 10],
        [0, 6, 4, 7, 5, 2, 8, 0, 10, 7, 13, 5, 15],
    ],
];

pub fn edge_transition_2<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    for node in tree.iter() {
        if let Some(cell_subcells) = tree.all_leaves(node) {
            for (facet, rows) in EDGES.iter().enumerate() {
                if let Some(neighbor) = node.facets[facet]
                    && let Some(face_nested) =
                        tree.orthants_all_leaves_on_facet(&tree.nodes[neighbor.into()], facet ^ 1)
                {
                    let face_subsubcells: [usize; LL] =
                        from_fn(|k| face_nested[k / L][k % L].into());
                    for &row in rows {
                        template(
                            facet,
                            row,
                            cell_subcells,
                            &face_subsubcells,
                            node,
                            center_nodes,
                            coordinates,
                            connectivity,
                            nodes_map,
                            tree,
                        )
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    facet: usize,
    row: [usize; 13],
    cell_subcells: &[U; N],
    face_subsubcells: &[usize; LL],
    node: &Node<D, M, N, T, U>,
    center_nodes: &[usize],
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
    tree: &Octree<T, U>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let [
        adjacent_facet,
        cell_a,
        cell_b,
        adjacent_cell_a,
        adjacent_cell_b,
        face_a,
        face_b,
        face_c,
        face_d,
        diag_face_a,
        diag_face_b,
        diag_face_c,
        diag_face_d,
    ] = row;
    if let Some(adjacent_cell) = node.facets[adjacent_facet]
        && let Some(adjacent_cell_subcells) = tree.all_leaves(&tree.nodes[adjacent_cell.into()])
        && let Some(diagonal_cell) = tree.nodes[adjacent_cell.into()].facets[facet]
        && let Some(diag_nested) =
            tree.orthants_all_leaves_on_facet(&tree.nodes[diagonal_cell.into()], facet ^ 1)
    {
        let diag_face_subsubcells: [usize; LL] = from_fn(|k| diag_nested[k / L][k % L].into());
        let length: Scalar = tree.nodes[face_subsubcells[face_a]].length.into();
        let offset = &facet_direction(facet) * length;
        // The four stitched nodes were created by the face template one sub-subcell
        // length inward of these leaf centers; look them up in `nodes_map`.
        let lookup = |center: usize| -> Option<usize> {
            let coordinate = &coordinates[center] - &offset;
            nodes_map
                .get(&[
                    (2.0 * coordinate[0]) as usize,
                    (2.0 * coordinate[1]) as usize,
                    (2.0 * coordinate[2]) as usize,
                ])
                .copied()
        };
        if let Some(node_1) = lookup(center_nodes[face_subsubcells[face_a]])
            && let Some(node_2) = lookup(center_nodes[face_subsubcells[face_b]])
            && let Some(node_3) = lookup(center_nodes[diag_face_subsubcells[diag_face_a]])
            && let Some(node_4) = lookup(center_nodes[diag_face_subsubcells[diag_face_b]])
        {
            connectivity.push([
                center_nodes[cell_subcells[cell_a].into()],
                center_nodes[cell_subcells[cell_b].into()],
                node_1,
                node_2,
                center_nodes[adjacent_cell_subcells[adjacent_cell_a].into()],
                center_nodes[adjacent_cell_subcells[adjacent_cell_b].into()],
                node_3,
                node_4,
            ]);
            connectivity.push([
                center_nodes[face_subsubcells[face_a]],
                center_nodes[face_subsubcells[face_b]],
                node_2,
                node_1,
                center_nodes[diag_face_subsubcells[diag_face_a]],
                center_nodes[diag_face_subsubcells[diag_face_b]],
                node_4,
                node_3,
            ]);
            connectivity.push([
                center_nodes[face_subsubcells[face_b]],
                node_2,
                node_4,
                center_nodes[diag_face_subsubcells[diag_face_b]],
                center_nodes[face_subsubcells[face_d]],
                center_nodes[cell_subcells[cell_a].into()],
                center_nodes[adjacent_cell_subcells[adjacent_cell_a].into()],
                center_nodes[diag_face_subsubcells[diag_face_d]],
            ]);
            connectivity.push([
                center_nodes[face_subsubcells[face_a]],
                center_nodes[diag_face_subsubcells[diag_face_a]],
                node_3,
                node_1,
                center_nodes[face_subsubcells[face_c]],
                center_nodes[diag_face_subsubcells[diag_face_c]],
                center_nodes[adjacent_cell_subcells[adjacent_cell_b].into()],
                center_nodes[cell_subcells[cell_b].into()],
            ]);
        }
    }
}
