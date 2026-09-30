//! Quasi-meshfree methods.

pub mod block;
mod discretization;

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, Model, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
};
pub use discretization::{Discretization, Support};
