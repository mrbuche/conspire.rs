pub(crate) mod dielectric;

use crate::{
    domain::diffusion::{NodalFields, NodalForces, NodalStiffnesses},
    units::ElectricPotential,
};

pub type NodalPotentials = NodalFields<ElectricPotential>;
pub type NodalForcesElectrical = NodalForces<ElectricPotential>;
pub type NodalStiffnessesElectrical = NodalStiffnesses<ElectricPotential>;
