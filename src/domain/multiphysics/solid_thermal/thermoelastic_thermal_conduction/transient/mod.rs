use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates, NodalVelocities,
        block::element::Elements,
        multiphysics::solid_thermal::{NodalForceSolidThermal, NodalForcesSolidThermal},
        thermal::NodalTemperatures,
    },
    math::Tensor,
};

pub trait ThermoelasticThermalConductionTransientElements
where
    Self: Elements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesSolidThermal, ElementModelError> {
        let mut nodal_forces = (0..nodal_coordinates.len())
            .map(|_| NodalForceSolidThermal::default())
            .collect();
        self.nodal_forces_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            &mut nodal_forces,
        )?;
        Ok(nodal_forces)
    }
}

impl<B> ThermoelasticThermalConductionTransientElements for Model<B, 3>
where
    B: ThermoelasticThermalConductionTransientElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_forces_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_forces,
        )
    }
}

impl<B1, B2> ThermoelasticThermalConductionTransientElements for Blocks<B1, B2>
where
    B1: ThermoelasticThermalConductionTransientElements,
    B2: ThermoelasticThermalConductionTransientElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_forces,
        )?;
        self.1.nodal_forces_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_forces,
        )
    }
}
