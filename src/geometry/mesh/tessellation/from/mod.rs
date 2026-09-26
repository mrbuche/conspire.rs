#[cfg(test)]
pub mod test;

mod grid;

use crate::geometry::{
    grid::Isosurface,
    mesh::{
        Connectivity, Mesh,
        tessellation::{D, Tessellation},
    },
};
use std::cell::OnceCell;

impl From<Isosurface> for Tessellation {
    fn from(surface: Isosurface) -> Self {
        let connectivities = vec![Connectivity::Triangular(surface.faces.into())];
        Tessellation::from(Mesh::from((connectivities, surface.vertices)))
    }
}

impl From<Mesh<D>> for Tessellation {
    fn from(mesh: Mesh<D>) -> Self {
        let normals = mesh.normals();
        Self {
            mesh,
            normals,
            bvh: OnceCell::new(),
            features: OnceCell::new(),
        }
    }
}
