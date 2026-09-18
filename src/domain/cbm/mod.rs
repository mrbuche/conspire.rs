//! Continuum bond methods

pub mod block;

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, Model, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{
        NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements,
        hyperelastic::HyperelasticElements,
    },
};
pub use block::node::Weighting;
