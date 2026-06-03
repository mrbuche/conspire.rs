use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, M, N},
            },
            node::Orthants,
        },
    },
    math::{Scalar, TensorVec},
};
use std::array::from_fn;

const SCALE_1: Scalar = 0.5;

const fn subcells_on_own_face(facet: usize) -> [usize; 4] {
    match facet {
        0 => [0, 2, 4, 6],
        1 => [1, 3, 5, 7],
        2 => [0, 1, 4, 5],
        3 => [2, 3, 6, 7],
        4 => [0, 1, 2, 3],
        5 => [4, 5, 6, 7],
        _ => panic!(),
    }
}

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
        if let Some(orthants) = node.orthants() {
            for facet in 0..M {
                // Only the cell's subcells on this facet are stitched, so they are
                // all this template needs as leaves; the off-face subcells are free
                // to carry their own structure (no whole-cell pairing assumed).
                if subcells_on_own_face(facet)
                    .iter()
                    .all(|&subcell| tree.nodes[orthants[subcell].into()].is_leaf())
                    && let Some(face_cell) = node.facets[facet]
                    && let Some(face_subsubcells) =
                        cell_subcells_contain_leaves(tree, face_cell.into(), facet)
                {
                    template(
                        orthants,
                        center_nodes,
                        nodes_map,
                        facet,
                        face_subsubcells,
                        tree,
                        connectivity,
                        coordinates,
                        node_index,
                    )
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template<T, U>(
    orthants: &Orthants<N, U>,
    cells_nodes: &[usize],
    nodes_map: &mut NodeMap<D>,
    facet: usize,
    face_subsubcells: [usize; 16],
    tree: &Octree<T, U>,
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    node_index: &mut usize,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let face_subcells = subcells_on_own_face(facet);
    let cell_subcells_face_nodes: [usize; 4] =
        from_fn(|i| cells_nodes[orthants[face_subcells[i]].into()]);
    let adjacent_exterior_nodes = [
        cells_nodes[face_subsubcells[1]],
        cells_nodes[face_subsubcells[4]],
        cells_nodes[face_subsubcells[7]],
        cells_nodes[face_subsubcells[13]],
        cells_nodes[face_subsubcells[14]],
        cells_nodes[face_subsubcells[11]],
        cells_nodes[face_subsubcells[8]],
        cells_nodes[face_subsubcells[2]],
    ];
    let adjacent_interior_nodes = [
        cells_nodes[face_subsubcells[3]],
        cells_nodes[face_subsubcells[6]],
        cells_nodes[face_subsubcells[12]],
        cells_nodes[face_subsubcells[9]],
    ];
    let (scale_1, scale_2) = translations(facet, &face_subsubcells, tree);
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
        cells_nodes,
        cell_subcells_face_nodes,
        facet,
        face_subsubcells,
        interior_nodes,
        exterior_nodes,
        connectivity,
    )
}

#[allow(clippy::too_many_arguments)]
fn connectivity_template(
    cells_nodes: &[usize],
    cell_subcells_face_nodes: [usize; 4],
    facet: usize,
    face_subsubcells: [usize; 16],
    interior_nodes: [usize; 4],
    exterior_nodes: [usize; 8],
    connectivity: &mut Vec<[usize; N]>,
) {
    match facet {
        0 | 3 | 4 => {
            connectivity.push([
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[9]],
                interior_nodes[0],
                interior_nodes[1],
                interior_nodes[2],
                interior_nodes[3],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[1]],
                cells_nodes[face_subsubcells[4]],
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[3]],
                exterior_nodes[0],
                exterior_nodes[1],
                interior_nodes[1],
                interior_nodes[0],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[7]],
                cells_nodes[face_subsubcells[13]],
                cells_nodes[face_subsubcells[12]],
                interior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[14]],
                cells_nodes[face_subsubcells[11]],
                interior_nodes[3],
                interior_nodes[2],
                exterior_nodes[4],
                exterior_nodes[5],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[2]],
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[8]],
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                exterior_nodes[6],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[0]],
                cells_nodes[face_subsubcells[1]],
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[2]],
                cell_subcells_face_nodes[0],
                exterior_nodes[0],
                interior_nodes[0],
                exterior_nodes[7],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[4]],
                cells_nodes[face_subsubcells[5]],
                cells_nodes[face_subsubcells[7]],
                cells_nodes[face_subsubcells[6]],
                exterior_nodes[1],
                cell_subcells_face_nodes[1],
                exterior_nodes[2],
                interior_nodes[1],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[13]],
                cells_nodes[face_subsubcells[15]],
                cells_nodes[face_subsubcells[14]],
                interior_nodes[2],
                exterior_nodes[3],
                cell_subcells_face_nodes[3],
                exterior_nodes[4],
            ]);
            connectivity.push([
                cells_nodes[face_subsubcells[8]],
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[11]],
                cells_nodes[face_subsubcells[10]],
                exterior_nodes[6],
                interior_nodes[3],
                exterior_nodes[5],
                cell_subcells_face_nodes[2],
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
                cell_subcells_face_nodes[0],
                cell_subcells_face_nodes[1],
                cell_subcells_face_nodes[3],
                cell_subcells_face_nodes[2],
            ]);
            connectivity.push([
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                interior_nodes[1],
                cell_subcells_face_nodes[1],
                cell_subcells_face_nodes[3],
                exterior_nodes[4],
                exterior_nodes[1],
            ]);
            connectivity.push([
                exterior_nodes[6],
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                cell_subcells_face_nodes[2],
                cell_subcells_face_nodes[0],
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
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[9]],
            ]);
            connectivity.push([
                exterior_nodes[0],
                exterior_nodes[1],
                interior_nodes[1],
                interior_nodes[0],
                cells_nodes[face_subsubcells[1]],
                cells_nodes[face_subsubcells[4]],
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[3]],
            ]);
            connectivity.push([
                interior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                cells_nodes[face_subsubcells[6]],
                cells_nodes[face_subsubcells[7]],
                cells_nodes[face_subsubcells[13]],
                cells_nodes[face_subsubcells[12]],
            ]);
            connectivity.push([
                interior_nodes[3],
                interior_nodes[2],
                exterior_nodes[4],
                exterior_nodes[5],
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[14]],
                cells_nodes[face_subsubcells[11]],
            ]);
            connectivity.push([
                exterior_nodes[7],
                interior_nodes[0],
                interior_nodes[3],
                exterior_nodes[6],
                cells_nodes[face_subsubcells[2]],
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[8]],
            ]);
            connectivity.push([
                cell_subcells_face_nodes[0],
                exterior_nodes[0],
                interior_nodes[0],
                exterior_nodes[7],
                cells_nodes[face_subsubcells[0]],
                cells_nodes[face_subsubcells[1]],
                cells_nodes[face_subsubcells[3]],
                cells_nodes[face_subsubcells[2]],
            ]);
            connectivity.push([
                exterior_nodes[1],
                cell_subcells_face_nodes[1],
                exterior_nodes[2],
                interior_nodes[1],
                cells_nodes[face_subsubcells[4]],
                cells_nodes[face_subsubcells[5]],
                cells_nodes[face_subsubcells[7]],
                cells_nodes[face_subsubcells[6]],
            ]);
            connectivity.push([
                interior_nodes[2],
                exterior_nodes[3],
                cell_subcells_face_nodes[3],
                exterior_nodes[4],
                cells_nodes[face_subsubcells[12]],
                cells_nodes[face_subsubcells[13]],
                cells_nodes[face_subsubcells[15]],
                cells_nodes[face_subsubcells[14]],
            ]);
            connectivity.push([
                exterior_nodes[6],
                interior_nodes[3],
                exterior_nodes[5],
                cell_subcells_face_nodes[2],
                cells_nodes[face_subsubcells[8]],
                cells_nodes[face_subsubcells[9]],
                cells_nodes[face_subsubcells[11]],
                cells_nodes[face_subsubcells[10]],
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
                cell_subcells_face_nodes[0],
                cell_subcells_face_nodes[1],
                cell_subcells_face_nodes[3],
                cell_subcells_face_nodes[2],
                exterior_nodes[0],
                exterior_nodes[1],
                exterior_nodes[4],
                exterior_nodes[5],
            ]);
            connectivity.push([
                cell_subcells_face_nodes[1],
                cell_subcells_face_nodes[3],
                exterior_nodes[4],
                exterior_nodes[1],
                exterior_nodes[2],
                exterior_nodes[3],
                interior_nodes[2],
                interior_nodes[1],
            ]);
            connectivity.push([
                cell_subcells_face_nodes[2],
                cell_subcells_face_nodes[0],
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
    face_subsubcells: &[usize; 16],
    tree: &Octree<T, U>,
) -> (Coordinate<D>, Coordinate<D>)
where
    T: Copy + Into<Scalar>,
    U: Copy + Into<usize>,
{
    let length: Scalar = tree.nodes[face_subsubcells[0]].length.into();
    // Push the new nodes into the cell, away from the shared face: toward +axis
    // for a low-side facet, toward -axis for a high-side facet.
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

/// Flattens the neighbor's sixteen sub-subcells on the shared face (its mirror
/// face, `facet ^ 1`) into template order — but only when every one of them is a
/// leaf. `orthants_leaves_on_facet` reads the tree structure directly, so a single
/// missing leaf short-circuits the whole template without any pairing assumption.
fn cell_subcells_contain_leaves<T, U>(
    tree: &Octree<T, U>,
    cell_index: usize,
    facet: usize,
) -> Option<[usize; 16]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let orthants_leaves_on_facet = tree.orthants_leaves_on_facet(&tree.nodes[cell_index], facet ^ 1);
    let mut subsubcells = [0; 16];
    let mut index = 0;
    for subcell in orthants_leaves_on_facet {
        for subsubcell in subcell? {
            subsubcells[index] = subsubcell?.into();
            index += 1;
        }
    }
    Some(subsubcells)
}
