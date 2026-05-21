pub mod base;
pub mod tessellation;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize, const I: usize, const M: usize, T> {
    connectivity: T,
    coordinates: Coordinates<D, I>,
}

pub type PrimitiveMesh<const D: usize, const I: usize, const M: usize, const N: usize, T> =
    Mesh<D, I, M, Vec<[T; N]>>;

pub type TriangularMesh<const I: usize, T> = PrimitiveMesh<3, I, 2, 3, T>;

// move to from/mod.rs?
impl<const D: usize, const I: usize, const M: usize, T> From<(T, Coordinates<D, I>)>
    for Mesh<D, I, M, T>
{
    fn from((connectivity, coordinates): (T, Coordinates<D, I>)) -> Self {
        Self {
            coordinates,
            connectivity,
        }
    }
}

// move to from/mod.rs?
impl<const D: usize, const I: usize, const M: usize, T> From<Mesh<D, I, M, T>>
    for (T, Coordinates<D, I>)
{
    fn from(mesh: Mesh<D, I, M, T>) -> Self {
        (mesh.connectivity, mesh.coordinates)
    }
}
