pub mod base;
pub mod from;
pub mod io;

use crate::geometry::{Coordinates, mesh::TriangularMesh};

pub struct Tessellation<const I: usize, T> {
    mesh: TriangularMesh<I, T>,
    normals: Coordinates<3, I>,
}
