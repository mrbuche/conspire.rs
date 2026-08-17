pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    math::{Current, TensorRank1Vec, TensorRank2SparseVec2D, TensorRank2SparseVec2DSymmetric},
    units::{Force, ForcePerLength, ForcePerVelocity},
};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, Current, Force>;
pub type NodalStiffnessesSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerLength>;
pub type NodalDampingsSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerVelocity>;
pub type NodalDampingsSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, ForcePerVelocity>;
pub type NodalStiffnessesSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, ForcePerLength>;
