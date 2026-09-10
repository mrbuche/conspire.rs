use crate::geometry::ntree::node::slot::Slot;
#[cfg(test)]
pub(crate) mod test;

mod star;

use super::{D, N};
use crate::geometry::{
    mesh::from::ntree::dualization::{NodeMap, Star},
    ntree::{Octree, node::cell::Cell},
};

pub(super) fn vertex_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Cell,
    U: Slot,
{
    tree.star(center_nodes, connectivity);
    star::template(tree, center_nodes, connectivity, nodes_map)
}
