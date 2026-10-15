//! Virtual element methods.

pub mod agglomerate;
pub mod block;
mod from;
pub mod solid;
pub mod thermal;

pub use crate::domain::{
    Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinatesHistory,
    NodalVelocitiesHistory,
    solid::{
        NodalForcesSolid,
        dynamics::{ElasticExplicitDynamics, ElasticImplicitDynamics},
    },
};

pub type NodalCoordinates = crate::domain::NodalCoordinates<3>;
pub type NodalReferenceCoordinates = crate::domain::NodalReferenceCoordinates<3>;
pub type NodalVelocities = crate::domain::NodalVelocities<3>;
