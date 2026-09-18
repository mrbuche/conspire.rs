//! Continuum bond methods

pub mod block;
pub(crate) mod node;

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, Model, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
};
pub use block::Cbm;
