pub mod capacity;
pub mod conduction;

use crate::{
    fem::block::element::diffusion::{DiffusiveFiniteElement, ElementNodalFields},
    mechanics::TemperatureGradients,
    units::Power,
};

pub type ElementNodalTemperatures<const D: usize> = ElementNodalFields<Power, D>;

pub trait ThermalFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: DiffusiveFiniteElement<Power, G, M, N, P>,
{
    fn temperature_gradients(
        &self,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> TemperatureGradients<G> {
        self.gradients(nodal_temperatures)
    }
}

impl<T, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalFiniteElement<G, M, N, P> for T
where
    T: DiffusiveFiniteElement<Power, G, M, N, P>,
{
}
