pub mod base;
pub mod from;
pub mod tessellation;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize, const I: usize, const M: usize, T> {
    connectivity: T,
    coordinates: Coordinates<D, I>,
}

pub type PrimitiveMesh<const D: usize, const I: usize, const M: usize, const N: usize, T> =
    Mesh<D, I, M, Vec<[T; N]>>;

pub type TriangularMesh<const I: usize, T> = PrimitiveMesh<3, I, 2, 3, T>;
