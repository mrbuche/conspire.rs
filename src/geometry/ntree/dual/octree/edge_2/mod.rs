use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, L, M, N, facet_direction},
            },
            node::Node,
        },
    },
    math::Scalar,
};
use std::array::from_fn;

const LL: usize = L * L;

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
        if node.is_leaf() {
            continue;
        }
        let node_subnodes = tree.leaves(node);
        for (facet, rows) in EDGES.iter().enumerate() {
            if let Some(neighbor) = node.facets[facet]
                && let Some(face_nested) =
                    tree.orthants_all_leaves_on_facet(&tree.nodes[neighbor.into()], facet ^ 1)
            {
                let face_subsubnodes: [usize; LL] = from_fn(|k| face_nested[k / L][k % L].into());
                for &row in rows {
                    template(
                        facet,
                        row,
                        &node_subnodes,
                        &face_subsubnodes,
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

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    facet: usize,
    row: [usize; 13],
    node_subnodes: &[Option<U>; N],
    face_subsubnodes: &[usize; LL],
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
        node_a,
        node_b,
        adjacent_node_a,
        adjacent_node_b,
        face_a,
        face_b,
        face_c,
        face_d,
        diag_face_a,
        diag_face_b,
        diag_face_c,
        diag_face_d,
    ] = row;
    if let Some(leaf_a) = node_subnodes[node_a]
        && let Some(leaf_b) = node_subnodes[node_b]
        && let Some(adjacent_node) = node.facets[adjacent_facet]
        && let Some(diagonal_node) = tree.nodes[adjacent_node.into()].facets[facet]
        && let Some(diag_nested) =
            tree.orthants_all_leaves_on_facet(&tree.nodes[diagonal_node.into()], facet ^ 1)
    {
        let adjacent_node_subnodes = tree.leaves(&tree.nodes[adjacent_node.into()]);
        if let Some(adjacent_leaf_a) = adjacent_node_subnodes[adjacent_node_a]
            && let Some(adjacent_leaf_b) = adjacent_node_subnodes[adjacent_node_b]
        {
            let cell_a = center_nodes[leaf_a.into()];
            let cell_b = center_nodes[leaf_b.into()];
            let adjacent_a = center_nodes[adjacent_leaf_a.into()];
            let adjacent_b = center_nodes[adjacent_leaf_b.into()];
            let diag_face_subsubnodes: [usize; LL] = from_fn(|k| diag_nested[k / L][k % L].into());
            let length: Scalar = tree.nodes[face_subsubnodes[face_a]].length.into();
            let offset = &facet_direction(facet) * length;
            let lookup = |center| -> Option<usize> {
                let coordinate = &coordinates[center] - &offset;
                nodes_map
                    .get(&[
                        (2.0 * coordinate[0]) as usize,
                        (2.0 * coordinate[1]) as usize,
                        (2.0 * coordinate[2]) as usize,
                    ])
                    .copied()
            };
            if let Some(node_1) = lookup(center_nodes[face_subsubnodes[face_a]])
                && let Some(node_2) = lookup(center_nodes[face_subsubnodes[face_b]])
                && let Some(node_3) = lookup(center_nodes[diag_face_subsubnodes[diag_face_a]])
                && let Some(node_4) = lookup(center_nodes[diag_face_subsubnodes[diag_face_b]])
            {
                connectivity.push([
                    cell_a, cell_b, node_1, node_2, adjacent_a, adjacent_b, node_3, node_4,
                ]);
                connectivity.push([
                    center_nodes[face_subsubnodes[face_a]],
                    center_nodes[face_subsubnodes[face_b]],
                    node_2,
                    node_1,
                    center_nodes[diag_face_subsubnodes[diag_face_a]],
                    center_nodes[diag_face_subsubnodes[diag_face_b]],
                    node_4,
                    node_3,
                ]);
                connectivity.push([
                    center_nodes[face_subsubnodes[face_b]],
                    node_2,
                    node_4,
                    center_nodes[diag_face_subsubnodes[diag_face_b]],
                    center_nodes[face_subsubnodes[face_d]],
                    cell_a,
                    adjacent_a,
                    center_nodes[diag_face_subsubnodes[diag_face_d]],
                ]);
                connectivity.push([
                    center_nodes[face_subsubnodes[face_a]],
                    center_nodes[diag_face_subsubnodes[diag_face_a]],
                    node_3,
                    node_1,
                    center_nodes[face_subsubnodes[face_c]],
                    center_nodes[diag_face_subsubnodes[diag_face_c]],
                    adjacent_b,
                    cell_b,
                ]);
            }
        }
    }
}
