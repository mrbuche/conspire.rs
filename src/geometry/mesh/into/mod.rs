#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Connectivity, Mesh, NodeSets, SideSets},
};
use std::vec::IntoIter;

impl<const D: usize> From<Mesh<D>> for (Connectivities, Coordinates<D>) {
    fn from(mesh: Mesh<D>) -> Self {
        (mesh.connectivities, mesh.coordinates)
    }
}

impl<const D: usize> From<Mesh<D>> for (Connectivities, Coordinates<D>, NodeSets, SideSets) {
    fn from(mesh: Mesh<D>) -> Self {
        (
            mesh.connectivities,
            mesh.coordinates,
            mesh.node_sets,
            mesh.side_sets,
        )
    }
}

impl<const D: usize> IntoIterator for Mesh<D> {
    type Item = Connectivity;
    type IntoIter = IntoIter<Connectivity>;
    fn into_iter(self) -> Self::IntoIter {
        self.connectivities.into_blocks().into_iter()
    }
}
