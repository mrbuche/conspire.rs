#[cfg(test)]
mod dynamics;
pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_plastic;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_plastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod loads;
pub mod mass;
pub mod modal;
pub mod time_scale;
pub mod viscoelastic;

pub use crate::domain::solid::dynamics::{
    Checked, Consistent, ElasticExplicitDynamics, ElasticImplicitDynamics, Execution, Lumped,
    MassChoice, Stability, Unchecked,
};

pub use crate::domain::solid::{
    NodalDampingsSolid, NodalDampingsSolidSymmetric, NodalForcesSolid, NodalStiffnessesSolid,
    NodalStiffnessesSolidSymmetric,
};
