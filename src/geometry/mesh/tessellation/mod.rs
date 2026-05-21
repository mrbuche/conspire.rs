pub mod from;
pub mod io;

use crate::geometry::{Coordinates, mesh::TriangularMesh};

pub struct Tessellation<const I: usize, T> {
    mesh: TriangularMesh<I, T>,
    normals: Coordinates<3, I>,
}

impl<const I: usize, T> Tessellation<I, T> {
    pub fn mesh(&self) -> &TriangularMesh<I, T> {
        &self.mesh
    }
    pub fn normals(&self) -> &Coordinates<3, I> {
        &self.normals
    }
}

// below should be more general that tessellations only

use std::{
    io::{Result as ResultIO},
    path::Path,
};

pub trait Write
{
    fn write<P: AsRef<Path>>(&self, path: P) -> ResultIO<()>;
}
