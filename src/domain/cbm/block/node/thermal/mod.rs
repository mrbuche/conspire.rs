pub mod capacity;
pub mod conduction;

use super::diffusion::DiffusiveNode;
use crate::{domain::thermal::NodalTemperatures, mechanics::TemperatureGradient, units::Power};

pub trait ThermalNode
where
    Self: DiffusiveNode<Power>,
{
    fn temperature_gradient(&self, nodal_temperatures: &NodalTemperatures) -> TemperatureGradient {
        self.gradient(nodal_temperatures)
    }
}

impl<T> ThermalNode for T where T: DiffusiveNode<Power> {}
