use crate::domain::{
    Blocks, ElementModelError, Model, NodalCoordinates,
    block::element::Elements,
    multiphysics::solid_thermal::{
        NodalForceSolidThermal, NodalForcesSolidThermal, NodalStiffnessesSolidThermal,
    },
    thermal::NodalTemperatures,
};
use crate::math::Tensor;

pub trait ThermoelasticThermalConductionElements
where
    Self: Elements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesSolidThermal, ElementModelError> {
        let mut nodal_forces = (0..nodal_coordinates.len())
            .map(|_| NodalForceSolidThermal::default())
            .collect::<NodalForcesSolidThermal>();
        self.nodal_forces_into(nodal_coordinates, nodal_temperatures, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesSolidThermal, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolidThermal::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_temperatures,
            &mut nodal_stiffnesses,
        )?;
        Ok(nodal_stiffnesses)
    }
}

impl<B> ThermoelasticThermalConductionElements for Model<B, 3>
where
    B: ThermoelasticThermalConductionElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)
    }
}

impl<B1, B2> ThermoelasticThermalConductionElements for Blocks<B1, B2>
where
    B1: ThermoelasticThermalConductionElements,
    B2: ThermoelasticThermalConductionElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)?;
        self.1
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)
    }
}
