pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::math::{TensorRank1Vec, TensorRank2Vec2D};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, 1>;
pub type NodalStiffnessesSolid<const D: usize> = TensorRank2Vec2D<D, 1, 1>;
