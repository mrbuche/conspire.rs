pub mod base;
pub mod from;
pub mod into;
pub mod read;
pub mod write;

use crate::geometry::{Coordinates, mesh::Mesh};

const D: usize = 3;
const N: usize = 3;

pub struct Tessellation<T> {
    mesh: Mesh<3, T>,
    normals: Coordinates<D>,
}
