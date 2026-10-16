pub(crate) mod capacity;
pub(crate) mod conduction;
pub(crate) mod time_scale;
pub(crate) mod transient;

use crate::{
    domain::diffusion::{NodalFields, NodalForces, NodalStiffnesses},
    units::Temperature,
};

pub type NodalTemperatures = NodalFields<Temperature>;
pub type NodalForcesThermal = NodalForces<Temperature>;
pub type NodalStiffnessesThermal = NodalStiffnesses<Temperature>;
