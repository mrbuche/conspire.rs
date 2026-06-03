#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Connectivity, Mesh},
};
use std::vec::IntoIter;

impl<const D: usize> From<Mesh<D>> for (Connectivities, Coordinates<D>) {
    fn from(mesh: Mesh<D>) -> Self {
        (mesh.connectivities, mesh.coordinates.into_members())
    }
}

impl<const D: usize> IntoIterator for Mesh<D> {
    type Item = Connectivity;
    type IntoIter = IntoIter<Connectivity>;
    fn into_iter(self) -> Self::IntoIter {
        self.connectivities.into_members().into_iter()
    }
}
