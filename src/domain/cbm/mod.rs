//! Continuum bond methods.

pub mod block;
#[cfg(test)]
mod dynamics;

pub mod mass {
    pub use crate::domain::solid::mass::{
        FixedLumpedMasses, InverseMass, LumpedMassElements, MassMatrix, NodalLumpedMasses,
    };
}

pub use crate::domain::{
    ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
    NodalCoordinatesHistory, NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory,
    Root,
    block::element::Elements,
    solid::{
        NodalDampingsSolid, NodalForcesSolid, NodalStiffnessesSolid, SolidElements,
        dynamics::ElasticDynamics,
        elastic::ElasticElements,
        elastic_hyperviscous::ElasticHyperviscousElements,
        elastic_viscoplastic::{ElasticViscoplasticBCs, ElasticViscoplasticElements},
        hyperelastic::HyperelasticElements,
        hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
        hyperviscoelastic::HyperviscoelasticElements,
        viscoelastic::ViscoelasticElements,
    },
};
pub use block::node::Weighting;
