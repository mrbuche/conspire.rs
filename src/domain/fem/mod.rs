//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
#[cfg(test)]
mod dynamics;
mod from;
pub mod mass;
pub mod solid;
pub mod thermal;
pub mod time_scale;

pub(crate) use crate::domain::nodal_coordinates;
pub use crate::domain::{
    Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, FirstOrderMinimize,
    FirstOrderRoot, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
    NodalCoordinatesHistory, NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory,
    ProvidesTangent, SecondOrderMinimize, SolverFor, ZerothOrderRoot, block::element::Elements,
};
