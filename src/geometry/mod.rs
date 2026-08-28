//! Geometry and meshing.

/// Bounding boxes.
pub mod bbox;

/// Bounding volume hierarchies.
pub mod bvh;

/// Computer-aided design.
pub mod cad;

/// Primitive and polyhedral meshes.
pub mod mesh;

/// Orthotrees, such as quadtrees and octrees.
pub mod ntree;

/// Regular grids of values.
pub mod grid;

/// Segmentations and related.
pub mod segmentation;

use crate::{
    math::{Reference, TensorRank1, TensorRank1List, TensorRank1RefVec, TensorRank1Vec},
    units::{Dimensionless, Length},
};

pub type Coordinate<const D: usize> = TensorRank1<D, Reference, Length>;
pub type Coordinates<const D: usize> = TensorRank1Vec<D, Reference, Length>;
pub type CoordinateList<const D: usize, const N: usize> = TensorRank1List<D, Reference, N, Length>;
pub type CoordinatesRef<'a, const D: usize> = TensorRank1RefVec<'a, D, Reference, Length>;
pub type Direction<const D: usize> = TensorRank1<D, Reference, Dimensionless>;
pub type Directions<const D: usize> = TensorRank1Vec<D, Reference, Dimensionless>;
pub type DirectionList<const D: usize, const N: usize> =
    TensorRank1List<D, Reference, N, Dimensionless>;
pub type DirectionsRef<'a, const D: usize> = TensorRank1RefVec<'a, D, Reference, Dimensionless>;
