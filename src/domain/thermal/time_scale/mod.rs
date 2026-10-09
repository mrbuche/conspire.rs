use crate::{
    domain::{
        Blocks, ElementModelError, Model, block::element::Elements, thermal::NodalTemperatures,
    },
    math::Quantity,
    units::Time,
};

/// Elements that report how fast their diffusion can be.
pub trait ThermalTimeScaleElements
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given temperatures, the reciprocal of
    /// the largest eigenvalue of the lumped heat capacity inverted against the conduction
    /// tangent, which bounds a stable explicit time step.
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> ThermalTimeScaleElements for Model<B, D>
where
    B: ThermalTimeScaleElements,
{
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks.fastest_diffusive_time_scale(nodal_temperatures)
    }
}

impl<B1, B2> ThermalTimeScaleElements for Blocks<B1, B2>
where
    B1: ThermalTimeScaleElements,
    B2: ThermalTimeScaleElements,
{
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_diffusive_time_scale(nodal_temperatures)?
            .min(self.1.fastest_diffusive_time_scale(nodal_temperatures)?))
    }
}
