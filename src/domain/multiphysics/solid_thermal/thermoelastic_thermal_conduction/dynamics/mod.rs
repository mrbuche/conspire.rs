use super::transient::ThermoelasticThermalConductionTransientElements;
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        factor::fixed_indices,
        solid::{
            NodalForcesSolid,
            mass::{InverseMass, MassMatrix},
        },
        thermal::{
            NodalForcesThermal, NodalTemperatures,
            capacity::{HeatCapacityMatrix, InverseHeatCapacity, NodalTemperatureRates},
            dynamics::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{ExplicitDynamicsCoupled, IntegrationError, Times},
        optimize::EqualityConstraint,
    },
    units::Time,
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
    NodalTemperaturesHistory,
    NodalTemperatureRatesHistory,
);

fn held<M, C>(
    masses: &M,
    capacities: &C,
    equality_constraint: EqualityConstraint,
    mut velocities: NodalVelocities<3>,
) -> Result<(M::Inverse, C::Inverse, NodalVelocities<3>), IntegrationError>
where
    M: MassMatrix<3>,
    C: HeatCapacityMatrix,
{
    let fixed = fixed_indices(
        equality_constraint,
        "explicit thermoelastic-thermal dynamics",
    )?;
    let fixed_coordinates = fixed
        .iter()
        .filter(|&&index| index % 4 < 3)
        .map(|&index| 3 * (index / 4) + index % 4)
        .collect::<Vec<_>>();
    let fixed_temperatures = fixed
        .iter()
        .filter(|&&index| index % 4 == 3)
        .map(|&index| index / 4)
        .collect::<Vec<_>>();
    let masses = masses
        .inverse(&fixed_coordinates)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    let capacities = capacities
        .inverse(&fixed_temperatures)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    fixed_coordinates
        .iter()
        .for_each(|&index| velocities[index / 3][index % 3] = Default::default());
    Ok((masses, capacities, velocities))
}

/// The motion and temperatures of a thermoelastic model that conducts heat.
pub trait ThermoelasticThermalConductionDynamics {
    #[expect(clippy::too_many_arguments)]
    fn nodal_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl InverseMass<3>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), ElementModelError>;
    /// Integrates the motion and temperatures of the model with an explicit coupled integrator.
    ///
    /// The masses and heat capacities may be lumped or consistent. Fixed degrees of freedom,
    /// whose indices are `4 * node + component` with the temperature as the last component, are
    /// held by giving them no velocity or rate of change, so that their initial values are the
    /// ones prescribed. Linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ThermoelasticThermalConductionDynamics for Model<B, 3>
where
    B: ThermoelasticThermalConductionTransientElements,
{
    fn nodal_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl InverseMass<3>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), ElementModelError> {
        let nodal_forces =
            self.nodal_forces(nodal_coordinates, nodal_velocities, nodal_temperatures)?;
        let internal_forces = nodal_forces.iter().map(|force| force.0.clone()).collect();
        let internal_heating = nodal_forces.iter().map(|force| force.1).collect();
        Ok((
            masses.nodal_accelerations(external_forces, &internal_forces),
            capacities.nodal_temperature_rates(external_heating, &internal_heating),
        ))
    }
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError> {
        let (masses, capacities, initial_velocities) =
            held(masses, capacities, equality_constraint, initial_velocities)?;
        integrator.integrate_coupled(
            |_,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                self.nodal_rates(
                    coordinates,
                    velocities,
                    temperatures,
                    external_forces,
                    external_heating,
                    &masses,
                    &capacities,
                )
                .map_err(|error| error.to_string())
            },
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
    }
}
