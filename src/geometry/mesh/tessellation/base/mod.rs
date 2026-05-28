#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{
        TriangularMesh,
        tessellation::{D, Tessellation},
    },
};

impl<T> Tessellation<T> {
    pub fn mesh(&self) -> &TriangularMesh<T> {
        &self.mesh
    }
    pub fn normals(&self) -> &Coordinates<D> {
        &self.normals
    }
}
