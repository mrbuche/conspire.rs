pub(super) mod base;
pub(super) mod cut;
pub(super) mod dual;
pub(super) mod from;
pub(super) mod into;
pub(super) mod read;
pub(super) mod sdf;
pub(super) mod write;

use crate::{
    geometry::{bvh::BoundingVolumeHierarchy, mesh::Mesh},
    math::TensorRank1Vec2D,
};
use std::cell::OnceCell;

const D: usize = 3;

type Normals = TensorRank1Vec2D<D, 0>;

pub struct Tessellation {
    mesh: Mesh<D>,
    normals: Normals,
    bvh: OnceCell<BoundingVolumeHierarchy<D>>,
}
