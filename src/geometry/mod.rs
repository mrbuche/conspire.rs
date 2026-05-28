mod bbox;
mod bvh;
mod mesh;
mod ntree;

#[cfg(feature = "netcdf")]
pub use self::mesh::exodus::WriteExodus;
pub use self::{
    bbox::{BoundingBox, BoundingBoxes, Unite as BoundingBoxUnite},
    bvh::BoundingVolumeHierarchy,
    mesh::{
        HexahedralMesh, Mesh, PrimitiveMesh, QuadrilateralMesh, TetrahedralMesh, TriangularMesh,
        tessellation::Tessellation,
    },
    ntree::{
        BinaryTree, Hexadecatree, Octree, Orthotree, Quadtree,
        balance::{Balance, Balancing},
        dual::Dualization,
        pair::Pairing,
    },
};
use std::path::Path;

use crate::math::{TensorRank1, TensorRank1List, TensorRank1RefVec, TensorRank1Vec};

pub type Coordinate<const D: usize> = TensorRank1<D, 0>;
pub type Coordinates<const D: usize> = TensorRank1Vec<D, 0>;
pub type CoordinateList<const D: usize, const N: usize> = TensorRank1List<D, 0, N>;
pub type CoordinatesRef<'a, const D: usize> = TensorRank1RefVec<'a, D, 0>;

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
