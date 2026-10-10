use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalReferenceCoordinates, block::element::Elements,
        thermal::NodalTemperatures,
    },
    math::Quantity,
    units::Time,
};

/// Elements that report how fast their diffusion can be.
pub trait ThermalTimeScaleElements<const D: usize>
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given temperatures, the reciprocal of
    /// the largest eigenvalue of the lumped heat capacity inverted against the conduction
    /// tangent, which bounds a stable explicit time step.
    fn fastest_diffusive_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> ThermalTimeScaleElements<D> for Model<B, D>
where
    B: ThermalTimeScaleElements<D>,
{
    fn fastest_diffusive_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks
            .fastest_diffusive_time_scale(reference_coordinates, nodal_temperatures)
    }
}

impl<B1, B2, const D: usize> ThermalTimeScaleElements<D> for Blocks<B1, B2>
where
    B1: ThermalTimeScaleElements<D>,
    B2: ThermalTimeScaleElements<D>,
{
    fn fastest_diffusive_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_diffusive_time_scale(reference_coordinates, nodal_temperatures)?
            .min(
                self.1
                    .fastest_diffusive_time_scale(reference_coordinates, nodal_temperatures)?,
            ))
    }
}
