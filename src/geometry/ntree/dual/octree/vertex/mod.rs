#[cfg(test)]
#[cfg(feature = "netcdf")]
pub(crate) mod test;

pub(crate) mod star;

use super::N;
use crate::geometry::ntree::{Octree, node::split::Split};
use std::ops::Add;

pub fn vertex_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Add<Output = T> + Copy + PartialOrd + Split + Into<usize>,
    U: Copy + Into<usize>,
{
    star::template(tree, center_nodes, connectivity)
}
