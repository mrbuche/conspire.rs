pub mod capacity;
pub mod conduction;

use crate::{
    fem::block::element::thermal::ElementNodalTemperatures as TetrahedronNodalTemperatures,
    mechanics::TemperatureGradient, units::Power,
    vem::block::element::diffusion::DiffusiveVirtualElement,
};

pub use crate::domain::thermal::NodalTemperatures as ElementNodalTemperatures;

pub trait ThermalVirtualElement
where
    Self: DiffusiveVirtualElement<Power>,
{
    fn temperature_gradient(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> TemperatureGradient {
        self.gradient(nodal_temperatures)
    }
    fn tetrahedra_temperatures(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Vec<TetrahedronNodalTemperatures<4>> {
        self.tetrahedra_fields(nodal_temperatures)
    }
}

impl<T> ThermalVirtualElement for T where T: DiffusiveVirtualElement<Power> {}
