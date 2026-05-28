#[cfg(test)]
mod test;

use crate::geometry::{Coordinates, Mesh};

impl<const D: usize, const M: usize, T> From<Mesh<D, M, T>> for (T, Coordinates<D>) {
    fn from(mesh: Mesh<D, M, T>) -> Self {
        (mesh.connectivity, mesh.coordinates)
    }
}
