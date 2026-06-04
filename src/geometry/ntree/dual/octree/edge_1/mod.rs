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
    math::{Scalar, TensorVec},
};

const EDGES: [(usize, usize, [usize; 10]); 12] = [
    (2, 1, [7, 13, 5, 15, 3, 7, 0, 4, 2, 6]),
    (2, 4, [1, 4, 0, 5, 2, 3, 4, 5, 6, 7]),
    (1, 3, [7, 13, 5, 15, 2, 6, 1, 5, 0, 4]),
    (1, 4, [1, 4, 0, 5, 0, 2, 5, 7, 4, 6]),
    (3, 0, [2, 8, 0, 10, 0, 4, 3, 7, 1, 5]),
    (3, 5, [11, 14, 10, 15, 4, 5, 2, 3, 0, 1]),
    (0, 2, [2, 8, 0, 10, 1, 5, 2, 6, 3, 7]),
    (0, 5, [11, 14, 10, 15, 5, 7, 0, 2, 1, 3]),
    (4, 3, [11, 14, 10, 15, 6, 7, 0, 1, 4, 5]),
    (4, 0, [2, 8, 0, 10, 4, 6, 1, 3, 5, 7]),
    (5, 2, [1, 4, 0, 5, 0, 1, 6, 7, 2, 3]),
    (5, 1, [7, 13, 5, 15, 1, 3, 4, 6, 0, 2]),
];

pub fn edge_transition_1<T, U>(
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
    for node in tree.iter() {
        for &(facet_m, facet_n, indices) in EDGES.iter() {
            template(
                facet_m,
                facet_n,
                indices,
                node,
                center_nodes,
                nodes_map,
                node_index,
                tree,
                connectivity,
                coordinates,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    facet_m: usize,
    facet_n: usize,
    indices: [usize; 10],
    node: &Node<D, M, N, T, U>,
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
    let [
        m_a,
        m_b,
        m_c,
        m_d,
        face_m_a,
        face_m_b,
        face_n_a,
        face_n_b,
        diag_a,
        diag_b,
    ] = indices;
    if let Some(neighbor_m) = node.facets[facet_m]
        && let Some(neighbor_n) = node.facets[facet_n]
        && let Some(neighbor_diag) = tree.nodes[neighbor_m.into()].facets[facet_n]
        && tree.orthants_all_leaves_on_facet(node, facet_n).is_some()
    {
        let face_m_leaves = tree.leaves(&tree.nodes[neighbor_m.into()]);
        let face_n_leaves = tree.leaves(&tree.nodes[neighbor_n.into()]);
        let diag_leaves = tree.leaves(&tree.nodes[neighbor_diag.into()]);
        let sub_subnodes = tree.orthants_leaves_on_facet(node, facet_m);
        let sub_subnode = |k: usize| sub_subnodes[k / L].and_then(|inner| inner[k % L]);
        if let Some(node_m_a) = sub_subnode(m_a)
            && let Some(node_m_b) = sub_subnode(m_b)
            && let Some(node_m_c) = sub_subnode(m_c)
            && let Some(node_m_d) = sub_subnode(m_d)
            && let Some(node_face_m_a) = face_m_leaves[face_m_a]
            && let Some(node_face_m_b) = face_m_leaves[face_m_b]
            && let Some(node_face_n_a) = face_n_leaves[face_n_a]
            && let Some(node_face_n_b) = face_n_leaves[face_n_b]
            && let Some(node_diag_a) = diag_leaves[diag_a]
            && let Some(node_diag_b) = diag_leaves[diag_b]
        {
            let center_m_a = center_nodes[node_m_a.into()];
            let center_m_b = center_nodes[node_m_b.into()];
            let center_m_c = center_nodes[node_m_c.into()];
            let center_m_d = center_nodes[node_m_d.into()];
            let face_m_a = center_nodes[node_face_m_a.into()];
            let face_m_b = center_nodes[node_face_m_b.into()];
            let face_n_a = center_nodes[node_face_n_a.into()];
            let face_n_b = center_nodes[node_face_n_b.into()];
            let diag_a = center_nodes[node_diag_a.into()];
            let diag_b = center_nodes[node_diag_b.into()];
            let length: Scalar = tree.nodes[node_m_a.into()].length.into();
            let offset_m = &facet_direction(facet_m) * length;
            let offset_n = &facet_direction(facet_n) * length;
            let base_a = coordinates[center_m_a].clone();
            let base_b = coordinates[center_m_b].clone();
            coordinates.push(&base_a + &offset_m);
            coordinates.push(&base_a + &offset_m + &offset_n);
            coordinates.push(&base_a + &offset_n);
            coordinates.push(&base_b + &offset_m);
            coordinates.push(&base_b + &offset_m + &offset_n);
            coordinates.push(&base_b + &offset_n);
            let new = *node_index;
            for k in 0..6 {
                let coordinate = &coordinates[new + k];
                nodes_map.insert(
                    [
                        (2.0 * coordinate[0]) as usize,
                        (2.0 * coordinate[1]) as usize,
                        (2.0 * coordinate[2]) as usize,
                    ],
                    new + k,
                );
            }
            *node_index += 6;
            connectivity.push([
                center_m_a,
                new,
                new + 1,
                new + 2,
                center_m_b,
                new + 3,
                new + 4,
                new + 5,
            ]);
            connectivity.push([
                new,
                face_m_a,
                diag_a,
                new + 1,
                new + 3,
                face_m_b,
                diag_b,
                new + 4,
            ]);
            connectivity.push([
                new + 1,
                diag_a,
                face_n_a,
                new + 2,
                new + 4,
                diag_b,
                face_n_b,
                new + 5,
            ]);
            connectivity.push([
                center_m_a,
                new + 2,
                new + 1,
                new,
                center_m_c,
                face_n_a,
                diag_a,
                face_m_a,
            ]);
            connectivity.push([
                center_m_b,
                new + 3,
                new + 4,
                new + 5,
                center_m_d,
                face_m_b,
                diag_b,
                face_n_b,
            ]);
        }
    }
}
