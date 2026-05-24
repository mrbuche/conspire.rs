mod bbox;
mod bvh;
mod mesh;

pub use self::{
    bbox::{BoundingBox, BoundingBoxes, Unite as BoundingBoxUnite},
    bvh::BoundingVolumeHierarchy,
    mesh::{Mesh, tessellation::Tessellation},
};
use std::path::Path;

use crate::math::{TensorRank1, TensorRank1List, TensorRank1RefVec, TensorRank1Vec};

pub type Coordinate<const D: usize, const I: usize> = TensorRank1<D, I>;
pub type Coordinates<const D: usize, const I: usize> = TensorRank1Vec<D, I>;
pub type CoordinateList<const D: usize, const I: usize, const N: usize> = TensorRank1List<D, I, N>;
pub type CoordinatesRef<'a, const D: usize, const I: usize> = TensorRank1RefVec<'a, D, I>;

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
