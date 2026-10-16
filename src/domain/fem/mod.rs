//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
pub mod electrical;
mod from;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

pub(crate) use crate::domain::nodal_coordinates;
pub use crate::domain::{
    Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Minimize, Model,
    NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates, NodalCoordinatesHistory,
    NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory, ProvidesTangent, Root,
    SolverFor, block::element::Elements,
};
