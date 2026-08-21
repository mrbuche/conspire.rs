#[cfg(test)]
#[cfg(feature = "netcdf")]
pub(crate) mod test;

mod star;

use super::{D, N};
use crate::geometry::ntree::{
    Octree,
    dual::{NodeMap, Star},
    node::cell::Cell,
};

pub(super) fn vertex_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) where
    T: Cell,
    U: Copy + Into<usize>,
{
    tree.star(center_nodes, connectivity);
    star::template(tree, center_nodes, connectivity, nodes_map)
}
