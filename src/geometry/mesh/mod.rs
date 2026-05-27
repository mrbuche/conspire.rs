#[cfg(feature = "netcdf")]
pub mod exodus;

pub mod base;
pub mod from;
pub mod into;
pub mod tessellation;
pub mod write;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize, const I: usize, const M: usize, T> {
    connectivity: T,
    coordinates: Coordinates<D, I>,
}

pub type PrimitiveMesh<const D: usize, const I: usize, const M: usize, const N: usize, T> =
    Mesh<D, I, M, Vec<[T; N]>>;

pub type HexahedralMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 8, T>;
pub type PyramidalMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 5, T>;
pub type TetrahedralMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 4, T>;
pub type WedgeMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 6, T>;

pub type QuadrilateralMesh<const D: usize, const I: usize, T> = PrimitiveMesh<D, I, 2, 4, T>;
pub type TriangularMesh<const I: usize, T> = PrimitiveMesh<3, I, 2, 3, T>;
