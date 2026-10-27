pub mod capacity;
pub mod conduction;
mod time_scale;

use crate::{
    constitutive::thermal::Thermal,
    fem::block::{
        diffusion::DiffusiveElements,
        element::thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    mechanics::TemperatureGradients,
    units::Power,
};

pub use crate::domain::thermal::NodalTemperatures;

pub trait ThermalElements<C, F, const G: usize, const M: usize, const N: usize>
where
    C: Thermal,
    F: ThermalFiniteElement<G, M, N>,
    Self: DiffusiveElements<Power, C, F, G, M, N>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperatures,
    ) -> ElementNodalTemperatures<N> {
        self.nodal_fields_element(element_connectivity, nodal_temperatures)
    }
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Vec<TemperatureGradients<G>> {
        self.gradients(nodal_temperatures)
    }
}

impl<T, C, F, const G: usize, const M: usize, const N: usize> ThermalElements<C, F, G, M, N> for T
where
    C: Thermal,
    F: ThermalFiniteElement<G, M, N>,
    T: DiffusiveElements<Power, C, F, G, M, N>,
{
}
