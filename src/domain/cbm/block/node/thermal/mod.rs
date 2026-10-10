pub mod capacity;
pub mod conduction;

use super::Node;
use crate::{domain::thermal::NodalTemperatures, mechanics::TemperatureGradient};

pub trait ThermalNode {
    fn temperature_gradient(&self, nodal_temperatures: &NodalTemperatures) -> TemperatureGradient;
}

impl ThermalNode for Node {
    fn temperature_gradient(&self, nodal_temperatures: &NodalTemperatures) -> TemperatureGradient {
        self.neighbors()
            .iter()
            .zip(self.gradient_vectors())
            .map(|(&neighbor, gradient_vector)| gradient_vector * nodal_temperatures[neighbor])
            .sum()
    }
}
