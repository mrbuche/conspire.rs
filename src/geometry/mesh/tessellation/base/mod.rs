#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    mesh::{TriangularMesh, tessellation::Tessellation},
};

impl<const I: usize, T> Tessellation<I, T> {
    pub fn mesh(&self) -> &TriangularMesh<I, T> {
        &self.mesh
    }
    pub fn normals(&self) -> &Coordinates<3, I> {
        &self.normals
    }
}
