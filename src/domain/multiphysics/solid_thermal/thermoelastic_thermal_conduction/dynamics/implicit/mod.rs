use super::{Solution, ThermoelasticThermalConductionDynamics, held};
use crate::{
    domain::{
        ElementModel, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalHeatingsSolidThermal, NodalStatesSolidThermal,
            NodalStiffnessesSolidThermal,
            thermoelastic_thermal_conduction::transient::ThermoelasticThermalConductionTransientElements,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, mass::InertialMass},
        thermal::{
            NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures,
            capacity::{InertialHeatCapacity, NodalTemperatureRates},
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Tensor, TensorTuple,
        integrate::{ImplicitDynamicsCoupled, IntegrationError},
        optimize::{EqualityConstraint, RootFinding},
    },
    units::{Rate, Time},
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

/// The motion and temperatures of a thermoelastic model that conducts heat, with implicit time
/// integration.
pub trait ThermoelasticThermalConductionImplicitDynamics {
    /// Integrates the motion and temperatures of the model with an implicit coupled integrator.
    ///
    /// The masses and heat capacities may be lumped or consistent, and are added to the sparse
    /// tangent, which is the exact derivative of the residual. Fixed degrees of freedom, whose
    /// indices are `4 * node + component` with the temperature as the last component, are held at
    /// their initial values, and linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ImplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalStatesSolidThermal,
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        solver: impl RootFinding<
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermal,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl InertialMass<3>,
        capacities: &impl InertialHeatCapacity,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ThermoelasticThermalConductionImplicitDynamics for Model<B, 3>
where
    B: ThermoelasticThermalConductionTransientElements,
{
    fn integrate(
        &self,
        integrator: &impl ImplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalStatesSolidThermal,
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        solver: impl RootFinding<
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermal,
        >,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl InertialMass<3>,
        capacities: &impl InertialHeatCapacity,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError> {
        let (inverse_masses, inverse_capacities, initial_velocities) = held(
            masses,
            capacities,
            equality_constraint.clone(),
            initial_velocities,
        )?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, 4, false);
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
                    &inverse_masses,
                    &inverse_capacities,
                )
                .map_err(|error| error.to_string())
            },
            |_,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             accelerations: &NodalAccelerations<3>,
             temperatures: &NodalTemperatures,
             rates: &NodalTemperatureRates| {
                let internal = self
                    .nodal_forces(coordinates, velocities, temperatures)
                    .map_err(|error| error.to_string())?;
                let inertial_forces = InertialMass::inertial_forces(masses, accelerations);
                let capacitive_heating = capacities.capacitive_heating(rates);
                Ok(internal
                    .into_iter()
                    .zip(inertial_forces)
                    .zip(capacitive_heating)
                    .zip(external_forces)
                    .zip(external_heating)
                    .map(
                        |((((internal, inertial_force), capacitive), external_force), external)| {
                            TensorTuple(
                                inertial_force + internal.0 - external_force,
                                capacitive + internal.1 - *external,
                            )
                        },
                    )
                    .collect())
            },
            |_,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             temperatures: &NodalTemperatures,
             dt,
             beta,
             gamma| {
                let (mut tangent, heatings) = self
                    .nodal_stiffnesses(coordinates, velocities, temperatures)
                    .map_err(|error| error.to_string())?;
                add_heatings(
                    &mut tangent,
                    &heatings,
                    Quantity::<Rate>::new(gamma / (beta * dt.value())),
                );
                let mut inertial = NodalStiffnessesSolid::<3>::zero(coordinates.len());
                masses.add_to_stiffnesses(1.0 / (beta * dt.value().powi(2)), &mut inertial);
                inertial.iter().enumerate().for_each(|(a, row)| {
                    row.entries().for_each(|(b, stiffness)| {
                        tangent[a][b] += TensorTuple(
                            TensorTuple(stiffness.clone(), Default::default()),
                            TensorTuple(Default::default(), Default::default()),
                        )
                    })
                });
                let mut capacitive = NodalStiffnessesThermal::zero(coordinates.len());
                capacities.add_to_stiffnesses(1.0 / dt.value(), &mut capacitive);
                capacitive.iter().enumerate().for_each(|(a, row)| {
                    row.entries().for_each(|(b, &stiffness)| {
                        tangent[a][b] += TensorTuple(
                            TensorTuple(Default::default(), Default::default()),
                            TensorTuple(Default::default(), stiffness),
                        )
                    })
                });
                Ok(tangent)
            },
            |coordinates: &NodalCoordinates<3>, temperatures: &NodalTemperatures| {
                coordinates
                    .iter()
                    .zip(temperatures.iter())
                    .map(|(coordinate, &temperature)| TensorTuple(coordinate.clone(), temperature))
                    .collect()
            },
            |states: &NodalStatesSolidThermal| {
                (
                    states.iter().map(|state| state.0.clone()).collect(),
                    states.iter().map(|state| state.1).collect(),
                )
            },
            solver,
            equality_constraint,
            Some(sparse),
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
    }
}
