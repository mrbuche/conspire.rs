pub mod conduction;

use crate::{
    fem::block::element::{Element, FiniteElement},
    math::{Tensor, TensorRank0List},
    mechanics::TemperatureGradients,
};

pub type ElementNodalTemperatures<const D: usize> = TensorRank0List<D>;

pub trait ThermalFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N>,
{
    fn temperature_gradients(
        &self,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> TemperatureGradients<G>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize> ThermalFiniteElement<G, M, N>
    for Element<G, N, O>
where
    Self: FiniteElement<G, M, N>,
{
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
