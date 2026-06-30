#[cfg(test)]
pub mod test;

mod grid;

use crate::geometry::mesh::{
    Mesh,
    tessellation::{D, Tessellation},
};
use std::cell::OnceCell;

impl From<Mesh<D>> for Tessellation {
    fn from(mesh: Mesh<D>) -> Self {
        let normals = mesh.normals();
        Self {
            mesh,
            normals,
            bvh: OnceCell::new(),
        }
    }
}
