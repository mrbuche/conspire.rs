pub(crate) mod dielectric;

use crate::{
    domain::diffusion::{NodalFields, NodalForces, NodalStiffnesses},
    units::Charge,
};

pub type NodalPotentials = NodalFields<Charge>;
pub type NodalForcesElectrical = NodalForces<Charge>;
pub type NodalStiffnessesElectrical = NodalStiffnesses<Charge>;
