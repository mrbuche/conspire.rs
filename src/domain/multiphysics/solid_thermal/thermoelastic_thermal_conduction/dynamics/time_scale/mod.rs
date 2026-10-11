use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::element::Elements, thermal::NodalTemperatures,
    },
    math::Quantity,
    units::Time,
};

/// Elements that report how fast their motion and their diffusion can be.
pub trait ThermoelasticThermalConductionTimeScaleElements
where
    Self: Elements,
{
    /// The shortest time scales among the elements at the given coordinates and temperatures,
    /// the reciprocals of the highest angular frequency of the motion and of the largest
    /// eigenvalue of the diffusion, in that order, which bound a stable explicit time step.
    ///
    /// The motion is that at the given temperatures, so it is slower than the adiabatic motion
    /// of the coupled system by the factor $`1 + 9\kappa\alpha^2 T/c`$ in stiffness, which is
    /// small for solids and absorbed by the safety factor of the time step.
    fn fastest_time_scales(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<(Quantity<Time>, Quantity<Time>), ElementModelError>;
}

impl<B> ThermoelasticThermalConductionTimeScaleElements for Model<B, 3>
where
    B: ThermoelasticThermalConductionTimeScaleElements,
{
    fn fastest_time_scales(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<(Quantity<Time>, Quantity<Time>), ElementModelError> {
        self.blocks.fastest_time_scales(
            reference_coordinates,
            nodal_coordinates,
            nodal_temperatures,
        )
    }
}

impl<B1, B2> ThermoelasticThermalConductionTimeScaleElements for Blocks<B1, B2>
where
    B1: ThermoelasticThermalConductionTimeScaleElements,
    B2: ThermoelasticThermalConductionTimeScaleElements,
{
    fn fastest_time_scales(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<(Quantity<Time>, Quantity<Time>), ElementModelError> {
        let first = self.0.fastest_time_scales(
            reference_coordinates,
            nodal_coordinates,
            nodal_temperatures,
        )?;
        let second = self.1.fastest_time_scales(
            reference_coordinates,
            nodal_coordinates,
            nodal_temperatures,
        )?;
        Ok((first.0.min(second.0), first.1.min(second.1)))
    }
}
