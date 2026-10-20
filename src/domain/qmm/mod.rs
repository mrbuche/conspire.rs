//! Quasi-meshfree methods.

pub mod block;
pub mod boundary {
    pub use crate::domain::boundary::{
        BoundaryConditions, Component, Condition, Conservative, ConservativeAt, Constrain,
        ConstrainAt, Direct, Displacement, Enforcement, Fixed, Layout, Left, Load, LoadAt,
        LoadStiffness, LoadStiffnessAt, Loads, NodalDof, NodalState, PowerAt, Prescribable,
        Prescribed, Right, Rows, Solid, SolidRate, Static, Timed, Velocity,
    };
}
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
