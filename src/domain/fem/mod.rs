//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
pub mod boundary {
    pub use crate::domain::boundary::{
        BoundaryConditions, Component, Concentration, Condition, Constrain, ConstrainAt, Coupled,
        Direct, Displacement, Enforcement, Field, Fixed, Layout, Left, NodalDof, NodalState,
        Potential, Prescribable, Prescribed, Right, Rows, Solid, SolidRate, SolidThermal, Static,
        Temperature, Timed, Velocity,
    };
}
pub mod chemical;
pub mod diffusion {
    pub use crate::domain::diffusion::{
        DiffusionElements, NodalFields, NodalForces, NodalStiffnesses,
    };
}
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
