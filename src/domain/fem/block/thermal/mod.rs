pub mod conduction;

use crate::{
    fem::block::{
        Block,
        element::thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::Vector,
    mechanics::TemperatureGradients,
};

pub type NodalTemperatures = Vector;

pub trait ThermalFiniteElementBlock<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
where
    F: ThermalFiniteElement<G, M, N, P>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperatures,
    ) -> ElementNodalTemperatures<N>;
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Vec<TemperatureGradients<G>>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ThermalFiniteElementBlock<C, F, G, M, N, P>
    for Block<C, F, G, M, N, P>
where
    F: ThermalFiniteElement<G, M, N, P>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperatures,
    ) -> ElementNodalTemperatures<N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_temperatures[node])
            .collect()
    }
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Vec<TemperatureGradients<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.temperature_gradients(
                    &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                )
            })
            .collect()
    }
}
