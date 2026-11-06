#[cfg(test)]
mod test;

pub(super) mod base;
pub(super) mod corefine;
pub(super) mod cut;
pub(super) mod features;
pub(super) mod from;
pub(super) mod into;
pub(super) mod read;
pub(super) mod sdf;
pub(super) mod trim;
pub(super) mod watertight;
pub(super) mod write;

use crate::{
    geometry::{Direction, bvh::BoundingVolumeHierarchy, mesh::Mesh},
    math::{Reference, TensorRank1Vec2D},
    units::Dimensionless,
};
use core::cell::OnceCell;
use features::Features;

const D: usize = 3;

const NORMALIZED_DIRECTIONS: [Direction<D>; 3] = [
    Direction::const_from([0.986187598618801, 0.13847260268289105, 0.09088101362060515]),
    Direction::const_from([0.09587682913992605, 0.9868622869851539, 0.13004022516323552]),
    Direction::const_from([0.12206546310376697, 0.08666652131910639, 0.9887309729141227]),
];

type Normals = TensorRank1Vec2D<D, Reference, Dimensionless>;

pub struct Tessellation {
    mesh: Mesh<D>,
    normals: Normals,
    bvh: OnceCell<BoundingVolumeHierarchy<D>>,
    features: OnceCell<Features>,
}
