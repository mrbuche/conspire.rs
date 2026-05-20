mod bbox;
mod bvh;

pub use self::{
    bbox::{BoundingBox, Unite as BoundingBoxUnite},
    bvh::BoundingVolumeHierarchy,
};

use crate::math::{TensorRank1, TensorRank1List, TensorRank1Vec};

pub type Coordinate<const D: usize, const I: usize> = TensorRank1<D, I>;
pub type Coordinates<const D: usize, const I: usize> = TensorRank1Vec<D, I>;
pub type CoordinateList<const D: usize, const I: usize, const N: usize> = TensorRank1List<D, I, N>;
