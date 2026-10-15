use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates, NodalVelocities,
        block::element::Elements,
        multiphysics::solid_thermal::{
            NodalForceSolidThermal, NodalForcesSolidThermal, NodalHeatingsSolidThermal,
            NodalStiffnessesSolidThermal,
        },
        thermal::{NodalStiffnessesThermal, NodalTemperatures, capacity::InertialHeatCapacity},
    },
    math::{Quantity, Scalar, Tensor, TensorTuple},
    units::Rate,
};

pub(crate) fn add_heatings(
    tangent: &mut NodalStiffnessesSolidThermal,
    heatings: &NodalHeatingsSolidThermal,
    velocity_rate: Quantity<Rate>,
) {
    heatings.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, heating)| {
            tangent[a][b] += TensorTuple(
                TensorTuple(Default::default(), Default::default()),
                TensorTuple(heating.clone() * velocity_rate, Default::default()),
            )
        })
    });
}

pub(crate) fn add_capacities(
    tangent: &mut NodalStiffnessesSolidThermal,
    capacities: &impl InertialHeatCapacity,
    weight: Scalar,
) {
    let mut capacitive = NodalStiffnessesThermal::zero(tangent.len());
    capacities.add_to_stiffnesses(weight, &mut capacitive);
    capacitive.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, &stiffness)| {
            tangent[a][b] += TensorTuple(
                TensorTuple(Default::default(), Default::default()),
                TensorTuple(Default::default(), stiffness),
            )
        })
    });
}

pub trait ThermoelasticThermalConductionRateElements
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
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
        nodal_heatings: &mut NodalHeatingsSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<(NodalStiffnessesSolidThermal, NodalHeatingsSolidThermal), ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolidThermal::zero(nodal_coordinates.len());
        let mut nodal_heatings = NodalHeatingsSolidThermal::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            &mut nodal_stiffnesses,
            &mut nodal_heatings,
        )?;
        Ok((nodal_stiffnesses, nodal_heatings))
    }
}

impl<B> ThermoelasticThermalConductionRateElements for Model<B, 3>
where
    B: ThermoelasticThermalConductionRateElements,
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
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
        nodal_heatings: &mut NodalHeatingsSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_stiffnesses,
            nodal_heatings,
        )
    }
}

impl<B1, B2> ThermoelasticThermalConductionRateElements for Blocks<B1, B2>
where
    B1: ThermoelasticThermalConductionRateElements,
    B2: ThermoelasticThermalConductionRateElements,
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
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
        nodal_heatings: &mut NodalHeatingsSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_stiffnesses,
            nodal_heatings,
        )?;
        self.1.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_velocities,
            nodal_temperatures,
            nodal_stiffnesses,
            nodal_heatings,
        )
    }
}
