use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, L, M, N},
            },
        },
    },
    math::{Scalar, TensorVec},
};
use std::array::from_fn;

const LL: usize = L * L;
const SCALE_1: Scalar = 0.5;

pub fn face_transition<T, U>(
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
        for facet in 0..M {
            if let [Some(s0), Some(s1), Some(s2), Some(s3)] = tree.leaves_on_facet(node, facet)
                && let Some(neighbor) = node.facets[facet]
                && let Some(neighbors) =
                    tree.subleaves_on_facet(&tree.nodes[neighbor.into()], facet ^ 1)
            {
                template(
                    [s0, s1, s2, s3],
                    center_nodes,
                    nodes_map,
                    facet,
                    neighbors,
                    tree,
                    connectivity,
                    coordinates,
                    node_index,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    leaves: [U; L],
    center_nodes: &[usize],
    nodes_map: &mut NodeMap<D>,
    facet: usize,
    neighbor_leaves: [[U; L]; L],
    tree: &Octree<T, U>,
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    node_index: &mut usize,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let neighbors = from_fn(|k| neighbor_leaves[k / L][k % L].into());
    let leaves_center_nodes: [usize; L] = from_fn(|i| center_nodes[leaves[i].into()]);
    let adjacent_exterior_nodes = [
        center_nodes[neighbors[1]],
        center_nodes[neighbors[4]],
        center_nodes[neighbors[7]],
        center_nodes[neighbors[13]],
        center_nodes[neighbors[14]],
        center_nodes[neighbors[11]],
        center_nodes[neighbors[8]],
        center_nodes[neighbors[2]],
    ];
    let adjacent_interior_nodes = [
        center_nodes[neighbors[3]],
        center_nodes[neighbors[6]],
        center_nodes[neighbors[12]],
        center_nodes[neighbors[9]],
    ];
    let (scale_1, scale_2) = translations(facet, &neighbors, tree);
    let interior_nodes = [
        *node_index,
        *node_index + 1,
        *node_index + 2,
        *node_index + 3,
    ];
    *node_index += 4;
    for &adjacent in adjacent_interior_nodes.iter() {
        let coordinate = &coordinates[adjacent] + &scale_1;
        coordinates.push(coordinate);
    }
    let mut exterior_nodes = [0; 8];
    for (exterior_node, &adjacent) in exterior_nodes
        .iter_mut()
        .zip(adjacent_exterior_nodes.iter())
    {
        let coordinate = &coordinates[adjacent] + &scale_2;
        let indices = [
            (2.0 * coordinate[0]) as usize,
            (2.0 * coordinate[1]) as usize,
            (2.0 * coordinate[2]) as usize,
        ];
        if let Some(&node_id) = nodes_map.get(&indices) {
            *exterior_node = node_id;
        } else {
            *exterior_node = *node_index;
            coordinates.push(coordinate);
            nodes_map.insert(indices, *node_index);
            *node_index += 1;
        }
    }
    connectivity_template(
        center_nodes,
        leaves_center_nodes,
        facet,
        neighbors,
        interior_nodes,
        exterior_nodes,
        connectivity,
    )
}

#[allow(clippy::too_many_arguments)]
fn connectivity_template(
    center_nodes: &[usize],
    leaves_center_nodes: [usize; L],
    facet: usize,
    neighbors: [usize; LL],
    interior_nodes: [usize; 4],
    exterior_nodes: [usize; 8],
    connectivity: &mut Vec<[usize; N]>,
) {
    match facet {
        0 | 3 | 4 => {
            connectivity.push([
                center_nodes[neighbors[3]],
                center_nodes[neighbors[6]],
                center_nodes[neighbors[12]],
                center_nodes[neighbors[9]],
                interior_nodes[0],
                interior_nodes[1],
                interior_nodes[2],
                interior_nodes[3],
            ]);
            connectivity.push([
                center_nodes[neighbors[1]],
                center_nodes[neighbors[4]],
                center_nodes[neighbors[6]],
                center_nodes[neighbors[3]],
                exterior_nodes[0],
                exterior_nodes[1],
                interior_nodes[1],
                interior_nodes[0],
            ]);
            connectivity.push([
                center_nodes[neighbors[6]],
                center_nodes[neighbors[7]],
                center_nodes[neighbors[13]],
                center_nodes[neighbors[12]],
                interior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
            ]);
            connectivity.push([
                center_nodes[neighbors[9]],
                center_nodes[neighbors[12]],
                center_nodes[neighbors[14]],
                center_nodes[neighbors[11]],
                interior_nodes[3],
                interior_nodes[2],
                exterior_nodes[4],
                exterior_nodes[5],
            ]);
            connectivity.push([
                center_nodes[neighbors[2]],
                center_nodes[neighbors[3]],
                center_nodes[neighbors[9]],
                center_nodes[neighbors[8]],
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                exterior_nodes[6],
            ]);
            connectivity.push([
                center_nodes[neighbors[0]],
                center_nodes[neighbors[1]],
                center_nodes[neighbors[3]],
                center_nodes[neighbors[2]],
                leaves_center_nodes[0],
                exterior_nodes[0],
                interior_nodes[0],
                exterior_nodes[7],
            ]);
            connectivity.push([
                center_nodes[neighbors[4]],
                center_nodes[neighbors[5]],
                center_nodes[neighbors[7]],
                center_nodes[neighbors[6]],
                exterior_nodes[1],
                leaves_center_nodes[1],
                exterior_nodes[2],
                interior_nodes[1],
            ]);
            connectivity.push([
                center_nodes[neighbors[12]],
                center_nodes[neighbors[13]],
                center_nodes[neighbors[15]],
                center_nodes[neighbors[14]],
                interior_nodes[2],
                exterior_nodes[3],
                leaves_center_nodes[3],
                exterior_nodes[4],
            ]);
            connectivity.push([
                center_nodes[neighbors[8]],
                center_nodes[neighbors[9]],
                center_nodes[neighbors[11]],
                center_nodes[neighbors[10]],
                exterior_nodes[6],
                interior_nodes[3],
                exterior_nodes[5],
                leaves_center_nodes[2],
            ]);
            connectivity.push([
                interior_nodes[0],
                interior_nodes[1],
                interior_nodes[2],
                interior_nodes[3],
                exterior_nodes[0],
                exterior_nodes[1],
                exterior_nodes[4],
                exterior_nodes[5],
            ]);
            connectivity.push([
                exterior_nodes[0],
                exterior_nodes[1],
                exterior_nodes[4],
                exterior_nodes[5],
                leaves_center_nodes[0],
                leaves_center_nodes[1],
                leaves_center_nodes[3],
                leaves_center_nodes[2],
            ]);
            connectivity.push([
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                interior_nodes[1],
                leaves_center_nodes[1],
                leaves_center_nodes[3],
                exterior_nodes[4],
                exterior_nodes[1],
            ]);
            connectivity.push([
                exterior_nodes[6],
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                leaves_center_nodes[2],
                leaves_center_nodes[0],
                exterior_nodes[0],
                exterior_nodes[5],
            ]);
        }
        1 | 2 | 5 => {
            connectivity.push([
                interior_nodes[0],
                interior_nodes[1],
                interior_nodes[2],
                interior_nodes[3],
                center_nodes[neighbors[3]],
                center_nodes[neighbors[6]],
                center_nodes[neighbors[12]],
                center_nodes[neighbors[9]],
            ]);
            connectivity.push([
                exterior_nodes[0],
                exterior_nodes[1],
                interior_nodes[1],
                interior_nodes[0],
                center_nodes[neighbors[1]],
                center_nodes[neighbors[4]],
                center_nodes[neighbors[6]],
                center_nodes[neighbors[3]],
            ]);
            connectivity.push([
                interior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                center_nodes[neighbors[6]],
                center_nodes[neighbors[7]],
                center_nodes[neighbors[13]],
                center_nodes[neighbors[12]],
            ]);
            connectivity.push([
                interior_nodes[3],
                interior_nodes[2],
                exterior_nodes[4],
                exterior_nodes[5],
                center_nodes[neighbors[9]],
                center_nodes[neighbors[12]],
                center_nodes[neighbors[14]],
                center_nodes[neighbors[11]],
            ]);
            connectivity.push([
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                exterior_nodes[6],
                center_nodes[neighbors[2]],
                center_nodes[neighbors[3]],
                center_nodes[neighbors[9]],
                center_nodes[neighbors[8]],
            ]);
            connectivity.push([
                leaves_center_nodes[0],
                exterior_nodes[0],
                interior_nodes[0],
                exterior_nodes[7],
                center_nodes[neighbors[0]],
                center_nodes[neighbors[1]],
                center_nodes[neighbors[3]],
                center_nodes[neighbors[2]],
            ]);
            connectivity.push([
                exterior_nodes[1],
                leaves_center_nodes[1],
                exterior_nodes[2],
                interior_nodes[1],
                center_nodes[neighbors[4]],
                center_nodes[neighbors[5]],
                center_nodes[neighbors[7]],
                center_nodes[neighbors[6]],
            ]);
            connectivity.push([
                interior_nodes[2],
                exterior_nodes[3],
                leaves_center_nodes[3],
                exterior_nodes[4],
                center_nodes[neighbors[12]],
                center_nodes[neighbors[13]],
                center_nodes[neighbors[15]],
                center_nodes[neighbors[14]],
            ]);
            connectivity.push([
                exterior_nodes[6],
                interior_nodes[3],
                exterior_nodes[5],
                leaves_center_nodes[2],
                center_nodes[neighbors[8]],
                center_nodes[neighbors[9]],
                center_nodes[neighbors[11]],
                center_nodes[neighbors[10]],
            ]);
            connectivity.push([
                exterior_nodes[0],
                exterior_nodes[1],
                exterior_nodes[4],
                exterior_nodes[5],
                interior_nodes[0],
                interior_nodes[1],
                interior_nodes[2],
                interior_nodes[3],
            ]);
            connectivity.push([
                leaves_center_nodes[0],
                leaves_center_nodes[1],
                leaves_center_nodes[3],
                leaves_center_nodes[2],
                exterior_nodes[0],
                exterior_nodes[1],
                exterior_nodes[4],
                exterior_nodes[5],
            ]);
            connectivity.push([
                leaves_center_nodes[1],
                leaves_center_nodes[3],
                exterior_nodes[4],
                exterior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                interior_nodes[1],
            ]);
            connectivity.push([
                leaves_center_nodes[2],
                leaves_center_nodes[0],
                exterior_nodes[0],
                exterior_nodes[5],
                exterior_nodes[6],
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
            ]);
        }
        _ => panic!(),
    }
}

fn translations<T, U>(
    facet: usize,
    neighbors: &[usize; LL],
    tree: &Octree<T, U>,
) -> (Coordinate<D>, Coordinate<D>)
where
    T: Copy + Into<Scalar>,
    U: Copy + Into<usize>,
{
    let length: Scalar = tree.nodes[neighbors[0]].length.into();
    match facet {
        0 => (
            Coordinate::const_from([SCALE_1 * length, 0.0, 0.0]),
            Coordinate::const_from([length, 0.0, 0.0]),
        ),
        1 => (
            Coordinate::const_from([-SCALE_1 * length, 0.0, 0.0]),
            Coordinate::const_from([-length, 0.0, 0.0]),
        ),
        2 => (
            Coordinate::const_from([0.0, SCALE_1 * length, 0.0]),
            Coordinate::const_from([0.0, length, 0.0]),
        ),
        3 => (
            Coordinate::const_from([0.0, -SCALE_1 * length, 0.0]),
            Coordinate::const_from([0.0, -length, 0.0]),
        ),
        4 => (
            Coordinate::const_from([0.0, 0.0, SCALE_1 * length]),
            Coordinate::const_from([0.0, 0.0, length]),
        ),
        5 => (
            Coordinate::const_from([0.0, 0.0, -SCALE_1 * length]),
            Coordinate::const_from([0.0, 0.0, -length]),
        ),
        _ => panic!(),
    }
}
