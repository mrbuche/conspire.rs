//! Virtual element methods.

pub mod block;
pub mod volume {
    pub use crate::domain::volume::{NodalVolumeElements, NodalVolumes};
}
pub mod boundary {
    pub use crate::domain::boundary::{
        BoundaryConditions, Component, Concentration, Condition, Conservative, ConservativeAt,
        Constrain, ConstrainAt, Coupled, Direct, Displacement, Enforcement, Field, Fixed, Layout,
        Left, Load, LoadAt, LoadStiffness, LoadStiffnessAt, Loads, NodalDof, NodalState, Potential,
        PowerAt, Prescribable, Prescribed, Right, Rows, Solid, SolidRate, SolidThermal, Static,
        Temperature, Timed, Velocity,
    };
}
pub mod chemical {
    pub mod diffusion {
        pub use crate::domain::chemical::diffusion::{
            ChemicalDiffusionElements, NodalConcentrations, NodalMolarFlowRates,
            NodalVolumetricFlowRates,
        };
    }
}
pub mod diffusion {
    pub use crate::domain::diffusion::loads::NodalSource;
    pub use crate::domain::diffusion::{
        DiffusionElements, NodalFields, NodalForces, NodalStiffnesses,
    };
}
pub mod electrical {
    pub mod conduction {
        pub use crate::domain::electrical::conduction::{
            ElectricalConductionElements, NodalConductances, NodalCurrents,
        };
    }
    pub mod dielectric {
        pub use crate::domain::electrical::dielectric::DielectricElements;
    }
}
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
