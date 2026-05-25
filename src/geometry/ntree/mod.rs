pub mod base;
pub mod error;
pub mod index;
pub mod leaf;

use crate::geometry::ntree::leaf::Leaves;

pub struct Orthotree<const D: usize, const N: usize, T, U> {
    leaves: Leaves<D, T, U>,
}

pub type Quadtree<T, U> = Orthotree<2, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 8, T, U>;
