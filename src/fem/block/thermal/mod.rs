pub mod conduction;

use crate::{
    fem::{
        NodalTemperatures, NodalTemperaturesBlock,
        block::{ElementBlock, element::ThermalFiniteElement},
    },
    mechanics::TemperatureGradients,
};

pub trait ThermalFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: ThermalFiniteElement<G, N>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> NodalTemperatures<N>;
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Vec<TemperatureGradients<G>>;
}

impl<C, F, const G: usize, const N: usize> ThermalFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    F: ThermalFiniteElement<G, N>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> NodalTemperatures<N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_temperatures[node])
            .collect()
    }
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Vec<TemperatureGradients<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.temperature_gradients(
                    &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                )
            })
            .collect()
    }
}
