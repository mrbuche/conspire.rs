pub(crate) mod capacity;
pub(crate) mod conduction;
pub(crate) mod dynamics;
pub(crate) mod time_scale;

use crate::{
    math::{QuantitySparseVec2D, QuantityVector},
    units::{Power, PowerPerTemperature, Temperature},
};

pub type NodalTemperatures = QuantityVector<Temperature>;
pub type NodalForcesThermal = QuantityVector<Power>;
pub type NodalStiffnessesThermal = QuantitySparseVec2D<PowerPerTemperature>;
