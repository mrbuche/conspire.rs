//! Quasi-meshfree methods.

pub mod block;
mod discretization;
#[cfg(test)]
mod dynamics;

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, Model, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
};
pub use discretization::{Discretization, Support};
