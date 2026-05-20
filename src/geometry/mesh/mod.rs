pub mod tessellation;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize, const I: usize, const M: usize, T> {
    coordinates: Coordinates<D, I>,
    connectivity: T,
}

impl<const D: usize, const I: usize, const M: usize, T> From<Mesh<D, I, M, T>>
    for (Coordinates<D, I>, T)
{
    fn from(mesh: Mesh<D, I, M, T>) -> Self {
        (mesh.coordinates, mesh.connectivity)
    }
}

pub type PrimitiveMesh<const D: usize, const I: usize, const M: usize, const N: usize, T> =
    Mesh<D, I, M, Vec<[T; N]>>;

pub type TriangularMesh<const I: usize, T> = PrimitiveMesh<3, I, 2, 3, T>;

// pub type PolygonalMesh<const I: usize, T> = Mesh<3, I, 2, Vec<Vec<T>>>;
// pub type PolyhedralMesh<const I: usize, T> = Mesh<3, I, 3, (Vec<Vec<T>>, Vec<Vec<T>>)>;
