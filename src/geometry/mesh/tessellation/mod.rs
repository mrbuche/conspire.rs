pub mod base;
pub mod from;
pub mod read;
pub mod write;

use crate::geometry::{Coordinates, mesh::TriangularMesh};

pub struct Tessellation<const I: usize, T> {
    mesh: TriangularMesh<I, T>,
    normals: Coordinates<3, I>,
}
