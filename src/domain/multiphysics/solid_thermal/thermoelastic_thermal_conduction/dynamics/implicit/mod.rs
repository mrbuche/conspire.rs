use super::{Solution, ThermoelasticThermalConductionDynamics, held};
use crate::{
    domain::{
        ElementModel, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{ConstrainAt, LoadAt, SolidThermal},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
            join_states, split_states,
            thermoelastic_thermal_conduction::rates::{
                ThermoelasticThermalConductionRateElements, add_capacities, add_heatings,
            },
        },
        solid::{NodalStiffnessesSolid, mass::InertialMass},
        thermal::{
            NodalTemperatures,
            capacity::{InertialHeatCapacity, NodalTemperatureRates},
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Tensor, TensorTuple,
        integrate::{ImplicitDynamicsCoupled, IntegrationError},
        optimize::RootFinding,
    },
    units::{Rate, Time},
};

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
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl InertialMass<3>,
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ThermoelasticThermalConductionImplicitDynamics for Model<B, 3>
where
    B: ThermoelasticThermalConductionRateElements,
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
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl InertialMass<3>,
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError> {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (inverse_masses, inverse_capacities, initial_velocities, motion) = held(
            masses,
            capacities,
            conditions,
            initial_time,
            &mut initial_coordinates,
            &mut initial_temperatures,
            initial_velocities,
            "implicit thermoelastic-thermal dynamics",
            Stepping::Implicit,
        )?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse =
            solver_from_neighbors(&neighbors, &motion.constraint_at(initial_time), 4, false);
        integrator.integrate_coupled(
            |time,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                self.nodal_rates(
                    coordinates,
                    velocities,
                    temperatures,
                    time,
                    applied_loads,
                    &inverse_masses,
                    &inverse_capacities,
                )
                .map_err(|error| error.to_string())
            },
            |time,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             accelerations: &NodalAccelerations<3>,
             temperatures: &NodalTemperatures,
             rates: &NodalTemperatureRates| {
                let mut internal = self
                    .nodal_forces(coordinates, velocities, temperatures)
                    .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(
                    time,
                    &join_states(coordinates, temperatures),
                    &mut internal,
                )?;
                let inertial_forces = InertialMass::inertial_forces(masses, accelerations);
                let capacitive_heating = capacities.capacitive_heating(rates);
                Ok(internal
                    .into_iter()
                    .zip(inertial_forces)
                    .zip(capacitive_heating)
                    .map(|((internal, inertial_force), capacitive)| {
                        TensorTuple(inertial_force + internal.0, capacitive + internal.1)
                    })
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
                add_capacities(&mut tangent, capacities, 1.0 / dt.value());
                Ok(tangent)
            },
            join_states,
            split_states,
            solver,
            |time| motion.constraint_at(time),
            Some(sparse),
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
    }
}
