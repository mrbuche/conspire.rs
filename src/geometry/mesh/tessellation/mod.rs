pub mod from;

use crate::geometry::{Coordinates, mesh::TriangularMesh};

pub struct Tessellation<const I: usize, T> {
    mesh: TriangularMesh<I, T>,
    normals: Coordinates<3, I>,
}
