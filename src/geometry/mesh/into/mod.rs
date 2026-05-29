#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates, Mesh,
    mesh::{Connectivities, MeshNew},
};

impl<const D: usize, const M: usize, T> From<Mesh<D, M, T>> for (T, Coordinates<D>) {
    fn from(mesh: Mesh<D, M, T>) -> Self {
        (mesh.connectivity, mesh.coordinates)
    }
}

impl<const D: usize, T> From<MeshNew<D, T>> for (Connectivities<T>, Coordinates<D>) {
    fn from(mesh: MeshNew<D, T>) -> Self {
        (mesh.connectivities, mesh.coordinates)
    }
}
