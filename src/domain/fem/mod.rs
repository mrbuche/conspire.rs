//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
#[cfg(feature = "fem")]
pub mod volume {
    pub use crate::domain::volume::{NodalVolumeElements, NodalVolumes};
}
#[cfg(feature = "fem")]
pub mod boundary {
    pub use crate::domain::boundary::{
        BoundaryConditions, Component, Concentration, Condition, Conservative, ConservativeAt,
        Constrain, ConstrainAt, Coupled, Direct, Displacement, Enforcement, Field, Fixed, Layout,
        Left, Load, LoadAt, LoadStiffness, LoadStiffnessAt, Loads, NodalDof, NodalState, Potential,
        PowerAt, Prescribable, Prescribed, Right, Rows, Solid, SolidRate, SolidThermal, Static,
        Temperature, Timed, Velocity,
    };
}
#[cfg(feature = "fem")]
pub mod chemical;
#[cfg(feature = "fem")]
pub mod diffusion {
    pub use crate::domain::diffusion::loads::NodalSource;
    pub use crate::domain::diffusion::{
        DiffusionElements, NodalFields, NodalForces, NodalStiffnesses,
    };
}
#[cfg(feature = "fem")]
pub mod electrical;
#[cfg(feature = "fem")]
pub mod flux;
#[cfg(feature = "fem")]
mod from;
#[cfg(feature = "fem")]
pub mod multiphysics;
#[cfg(feature = "fem")]
pub mod solid;
#[cfg(feature = "fem")]
pub mod thermal;

#[cfg(feature = "fem")]
pub(crate) use crate::domain::nodal_coordinates;
#[cfg(feature = "fem")]
pub use crate::domain::{
    Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Minimize, Model,
    NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates, NodalCoordinatesHistory,
    NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory, ProvidesTangent, Root,
    SolverFor, block::element::Elements,
};
