pub mod capacity;
pub mod conduction;

use super::diffusion::DiffusiveNode;
use crate::{
    domain::thermal::NodalTemperatures, mechanics::TemperatureGradient, units::Temperature,
};

pub trait ThermalNode
where
    Self: DiffusiveNode<Temperature>,
{
    fn temperature_gradient(&self, nodal_temperatures: &NodalTemperatures) -> TemperatureGradient {
        self.gradient(nodal_temperatures)
    }
}

impl<T> ThermalNode for T where T: DiffusiveNode<Temperature> {}
