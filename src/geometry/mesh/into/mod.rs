#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Mesh},
};

impl<const D: usize> From<Mesh<D>> for (Connectivities, Coordinates<D>) {
    fn from(mesh: Mesh<D>) -> Self {
        (mesh.connectivities, mesh.coordinates)
    }
}
