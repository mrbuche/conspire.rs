#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        TriangularMesh,
        tessellation::{D, N, Tessellation},
    },
};

impl<const I: usize, T> From<Tessellation<I, T>> for TriangularMesh<I, T> {
    fn from(tessellation: Tessellation<I, T>) -> Self {
        tessellation.mesh
    }
}

impl<const I: usize, T> From<Tessellation<I, T>> for (Vec<[T; N]>, Coordinates<D, I>) {
    fn from(tessellation: Tessellation<I, T>) -> Self {
        (
            tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
        )
    }
}

impl<const I: usize, T> From<Tessellation<I, T>>
    for (Vec<[T; N]>, Coordinates<D, I>, Coordinates<D, I>)
{
    fn from(tessellation: Tessellation<I, T>) -> Self {
        (
            tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
            tessellation.normals,
        )
    }
}
