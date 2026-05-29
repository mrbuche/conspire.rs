#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        Mesh,
        tessellation::{D, Tessellation},
    },
};

impl<T> Tessellation<T> {
    pub fn mesh(&self) -> &Mesh<3, T> {
        &self.mesh
    }
    pub fn normals(&self) -> &Coordinates<D> {
        &self.normals
    }
}
