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
    units::Temperature,
};

pub use crate::domain::thermal::NodalTemperatures;

pub trait ThermalElements<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Thermal,
    F: ThermalFiniteElement<G, M, N, P>,
    Self: DiffusiveElements<Temperature, C, F, G, M, N, P>,
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

impl<T, C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalElements<C, F, G, M, N, P> for T
where
    C: Thermal,
    F: ThermalFiniteElement<G, M, N, P>,
    T: DiffusiveElements<Temperature, C, F, G, M, N, P>,
{
}
