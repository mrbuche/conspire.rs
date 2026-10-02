//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
pub mod dynamics;
mod from;
pub mod mass;
pub mod solid;
pub mod thermal;

pub(crate) use crate::domain::nodal_coordinates;
pub use crate::domain::{
    Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, FirstOrderMinimize,
    FirstOrderRoot, Model, NodalAccelerations, NodalCoordinates, NodalCoordinatesHistory,
    NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory, ProvidesTangent,
    SecondOrderMinimize, SolverFor, ZerothOrderRoot, block::element::Elements,
};
