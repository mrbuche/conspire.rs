pub mod triangular;

use crate::geometry::{Coordinates, mesh::Mesh};

pub struct Tessellation<const D: usize, const I: usize, const M: usize, T> {
    mesh: Mesh<D, I, M, T>,
    normals: Coordinates<D, I>,
}
