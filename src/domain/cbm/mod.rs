//! Continuum bond methods.

pub mod block;

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, Model, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{
        NodalDampingsSolid, NodalForcesSolid, NodalStiffnessesSolid, SolidElements,
        elastic::ElasticElements, elastic_hyperviscous::ElasticHyperviscousElements,
        hyperelastic::HyperelasticElements, hyperviscoelastic::HyperviscoelasticElements,
        viscoelastic::ViscoelasticElements,
    },
};
pub use block::node::Weighting;
