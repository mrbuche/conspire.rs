mod transition_1;
mod transition_2;
mod transition_3;
mod transition_4;

use super::{D, N};
use crate::{
    geometry::{
        Coordinates,
        ntree::{Octree, dual::NodeMap},
    },
    math::Scalar,
};

pub fn edge_transitions<T, U>(
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
    transition_1::edge_transition_1(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    transition_3::edge_transition_3(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    transition_2::edge_transition_2(tree, center_nodes, coordinates, connectivity, nodes_map);
    transition_4::edge_transition_4(tree, center_nodes, coordinates, connectivity, nodes_map);
}
