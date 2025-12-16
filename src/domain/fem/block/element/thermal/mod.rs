pub mod conduction;

use crate::{
    fem::block::element::Element,
    math::{Tensor, TensorRank0List},
    mechanics::TemperatureGradients,
};

pub type ElementNodalTemperatures<const D: usize> = TensorRank0List<D>;

pub trait ThermalFiniteElement<const G: usize, const N: usize> {
    fn temperature_gradients(
        &self,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> TemperatureGradients<G>;
}

impl<const G: usize, const N: usize> ThermalFiniteElement<G, N> for Element<G, N> {
    fn temperature_gradients(
        &self,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> TemperatureGradients<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_temperatures
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_temperature, gradient_vector)| gradient_vector * nodal_temperature)
                    .sum()
            })
            .collect()
    }
}
