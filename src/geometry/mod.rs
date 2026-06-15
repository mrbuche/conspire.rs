//! Geometry library.

/// Bounding boxes.
pub mod bbox;

/// Bounding volume hierarchies.
pub mod bvh;

/// Primitive and polyhedral meshes.
pub mod mesh;

/// Orthotrees, such as quadtrees and octrees.
pub mod ntree;

/// Regular grids of values.
pub mod grid;

use crate::math::{TensorRank1, TensorRank1List, TensorRank1RefVec, TensorRank1Vec};

pub type Coordinate<const D: usize> = TensorRank1<D, 0>;
pub type Coordinates<const D: usize> = TensorRank1Vec<D, 0>;
pub type CoordinateList<const D: usize, const N: usize> = TensorRank1List<D, 0, N>;
pub type CoordinatesRef<'a, const D: usize> = TensorRank1RefVec<'a, D, 0>;
