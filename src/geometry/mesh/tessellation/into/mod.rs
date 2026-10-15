#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivities, Mesh,
        tessellation::{D, Normals, Tessellation},
    },
};

impl From<Tessellation> for Mesh<D> {
    fn from(tessellation: Tessellation) -> Self {
        tessellation.mesh
    }
}

impl From<Tessellation> for (Connectivities, Coordinates<D>, Normals) {
    fn from(tessellation: Tessellation) -> Self {
        (
            tessellation.mesh.connectivities,
            tessellation.mesh.coordinates,
            tessellation.normals,
        )
    }
}
