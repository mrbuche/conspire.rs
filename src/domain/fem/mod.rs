//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
pub mod feti;
mod from;
pub mod solid;
pub mod thermal;

pub(crate) use crate::domain::nodal_coordinates;
pub use crate::domain::{
    Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, FirstOrderMinimize,
    FirstOrderRoot, Model, NodalCoordinates, NodalCoordinatesHistory, NodalReferenceCoordinates,
    NodalVelocities, NodalVelocitiesHistory, ProvidesTangent, SecondOrderMinimize, SolverFor,
    ZerothOrderRoot, block::element::Elements,
};
