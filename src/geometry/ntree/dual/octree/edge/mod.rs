mod transition_1;
mod transition_2;
mod transition_3;
mod transition_4;
mod transition_5;

use super::{D, N};
use crate::{
    geometry::{
        Coordinates,
        ntree::{Octree, balance::Balancing, dual::NodeMap},
    },
    math::Scalar,
};

#[allow(clippy::too_many_arguments)]
pub fn edge_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
    balancing: Balancing,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    transition_1::template(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    transition_3::template(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    transition_2::template(tree, center_nodes, coordinates, connectivity, nodes_map);
    transition_4::template(tree, center_nodes, coordinates, connectivity, nodes_map);
    if matches!(balancing, Balancing::Weak) {
        transition_5::template(
            tree,
            center_nodes,
            coordinates,
            connectivity,
            node_index,
            nodes_map,
        );
    }
}
