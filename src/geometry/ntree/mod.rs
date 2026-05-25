pub mod balance;
pub mod error;
pub mod from;
pub mod into;
pub mod leaf;
pub mod subdivide;

use crate::geometry::ntree::leaf::Leaves;

pub struct Orthotree<const D: usize, const N: usize, T, U> {
    leaves: Leaves<D, T, U>,
}

pub type Quadtree<T, U> = Orthotree<2, 4, T, U>;
pub type Octree<T, U> = Orthotree<3, 8, T, U>;
