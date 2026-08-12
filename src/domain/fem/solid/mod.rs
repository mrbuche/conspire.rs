use crate::math::{Current, Time};
pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::math::{TensorRank1Vec, TensorRank2SparseVec2D, TensorRank2SparseVec2DSymmetric};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, Current>;
pub type NodalStiffnessesSolid<const D: usize> = TensorRank2SparseVec2D<D, Current, Current>;
pub type NodalDampingsSolid<const D: usize> = TensorRank2SparseVec2D<D, Current, Current, Time>;
pub type NodalDampingsSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, Time>;
pub type NodalStiffnessesSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current>;
