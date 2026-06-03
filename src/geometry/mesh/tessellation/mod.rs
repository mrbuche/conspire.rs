pub mod base;
pub mod dual;
pub mod from;
pub mod into;
pub mod read;
pub mod sdf;
pub mod write;

use crate::{geometry::mesh::Mesh, math::TensorRank1Vec2D};

const D: usize = 3;

type Normals = TensorRank1Vec2D<D, 0>;

pub struct Tessellation {
    mesh: Mesh<D>,
    normals: Normals,
}
