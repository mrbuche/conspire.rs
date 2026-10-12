//! Quasi-meshfree methods.

pub mod block;
mod discretization;
#[cfg(test)]
mod dynamics;
pub mod mass {
    pub use crate::domain::solid::mass::{
        ConsistentMassElements, InertialMass, InverseMass, LumpedMassElements, MassMatrix,
        NodalLumpedMasses, NodalMasses,
    };
}

pub use crate::domain::{
    ElementModelError, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities, Root,
    block::element::Elements,
    solid::{
        NodalForcesSolid, NodalStiffnessesSolid, SolidElements,
        dynamics::{ElasticExplicitDynamics, ElasticImplicitDynamics},
        elastic::ElasticElements,
    },
};
pub use discretization::{Discretization, Support};
