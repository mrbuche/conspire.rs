#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        Mesh,
        tessellation::{D, N, Tessellation},
    },
};

impl<T> From<Tessellation<T>> for Mesh<3, T> {
    fn from(tessellation: Tessellation<T>) -> Self {
        tessellation.mesh
    }
}

impl<T> From<Tessellation<T>> for (Vec<[T; N]>, Coordinates<D>) {
    fn from(tessellation: Tessellation<T>) -> Self {
        (
            todo!(), // tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
        )
    }
}

impl<T> From<Tessellation<T>> for (Vec<[T; N]>, Coordinates<D>, Coordinates<D>) {
    fn from(tessellation: Tessellation<T>) -> Self {
        (
            todo!(), // tessellation.mesh.connectivity,
            tessellation.mesh.coordinates,
            tessellation.normals,
        )
    }
}
