pub mod base;
pub mod from;
pub mod into;
pub mod read;
pub mod write;

use crate::geometry::{Coordinates, mesh::TriangularMesh};

const D: usize = 3;
const N: usize = 3;

pub struct Tessellation<const I: usize, T> {
    mesh: TriangularMesh<I, T>,
    normals: Coordinates<D, I>,
}
