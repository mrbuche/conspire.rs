#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        TriangularMesh,
        tessellation::{D, N, Tessellation},
    },
};

impl<T> From<Tessellation<T>> for TriangularMesh<T> {
    fn from(tessellation: Tessellation<T>) -> Self {
        tessellation.mesh
    }
}

impl<T> From<Tessellation<T>> for (Vec<[T; N]>, Coordinates<D>) {
    fn from(tessellation: Tessellation<T>) -> Self {
        (
            tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
        )
    }
}

impl<T> From<Tessellation<T>> for (Vec<[T; N]>, Coordinates<D>, Coordinates<D>) {
    fn from(tessellation: Tessellation<T>) -> Self {
        (
            tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
            tessellation.normals,
        )
    }
}
