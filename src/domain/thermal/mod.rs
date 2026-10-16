pub(crate) mod capacity;
pub(crate) mod conduction;
pub(crate) mod time_scale;
pub(crate) mod transient;

use crate::{
    domain::diffusion::{NodalFields, NodalForces, NodalStiffnesses},
    units::Power,
};

pub type NodalTemperatures = NodalFields<Power>;
pub type NodalForcesThermal = NodalForces<Power>;
pub type NodalStiffnessesThermal = NodalStiffnesses<Power>;
