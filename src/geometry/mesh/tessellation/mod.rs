pub(super) mod base;
pub(super) mod cut;
pub(super) mod features;
pub(super) mod from;
pub(super) mod into;
pub(super) mod read;
pub(super) mod sdf;
pub(super) mod solid;
pub(super) mod trim;
pub(super) mod write;

use crate::{
    geometry::{bvh::BoundingVolumeHierarchy, mesh::Mesh},
    math::{Reference, TensorRank1Vec2D},
    units::Dimensionless,
};
use features::Features;
use std::cell::OnceCell;

const D: usize = 3;

type Normals = TensorRank1Vec2D<D, Reference, Dimensionless>;

pub struct Tessellation {
    mesh: Mesh<D>,
    normals: Normals,
    bvh: OnceCell<BoundingVolumeHierarchy<D>>,
    features: OnceCell<Features>,
}
