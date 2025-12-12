pub mod conduction;

use crate::{
    fem::{NodalTemperatures, block::element::Element},
    math::Tensor,
    mechanics::TemperatureGradients,
};
use std::fmt::Debug;

pub trait ThermalFiniteElement<const G: usize, const N: usize>
where
    Self: Debug,
{
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> TemperatureGradients<G>;
}

impl<const G: usize, const N: usize> ThermalFiniteElement<G, N> for Element<G, N>
where
    Self: Debug,
{
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> TemperatureGradients<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_temperatures
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_temperature, gradient_vector)| gradient_vector * nodal_temperature)
                    .sum()
            })
            .collect()
    }
}
