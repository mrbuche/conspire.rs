use super::{
    motion::Motion,
    rates::{ThermoelasticThermalConductionRateElements, add_capacities, add_heatings},
};
use crate::{
    domain::{
        ElementModel, Model, NodalCoordinates, NodalCoordinatesHistory, NodalVelocities,
        NodalVelocitiesHistory, Stepping,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{ConstrainAt, LoadAt, SolidThermal},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalRatesSolidThermal, NodalRatesSolidThermalHistory,
            NodalStatesSolidThermal, NodalStatesSolidThermalHistory, NodalStiffnessesSolidThermal,
            join_rates, join_states, split_rates, split_states,
        },
        thermal::{
            NodalTemperatures,
            capacity::{InertialHeatCapacity, InverseHeatCapacity},
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Tensor, TensorTuple,
        integrate::{ImplicitRates, IntegrationError, Times},
        optimize::RootFinding,
    },
    units::{Rate, Time},
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalTemperaturesHistory,
    NodalTemperatureRatesHistory,
);

/// The coordinates and temperatures of a thermoelastic model that conducts heat, without
/// inertia, with implicit time integration.
pub trait ThermoelasticThermalConductionQuasistatic {
    /// Integrates the coordinates and temperatures of the model with an implicit integrator of
    /// first-order systems.
    ///
    /// Equilibrium is algebraic in the coordinates, so the model has no mass and the velocities
    /// are the rates of the integrator, such as the backward difference of a backward Euler
    /// step, and the thermoelastic heating follows from them. The initial coordinates must be in
    /// equilibrium with the initial temperatures and external forces, and the velocities at the
    /// initial time are taken to be zero. The heat capacities may be lumped or consistent, and
    /// are added to the sparse tangent. Fixed degrees of freedom, whose indices are `4 * node +
    /// component` with the temperature as the last component, are held at their initial values,
    /// and linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ImplicitRates<
            NodalStatesSolidThermal,
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermalHistory,
            NodalRatesSolidThermalHistory,
        >,
        solver: impl RootFinding<
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermal,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalTemperatures),
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ThermoelasticThermalConductionQuasistatic for Model<B, 3>
where
    B: ThermoelasticThermalConductionRateElements,
{
    fn integrate(
        &self,
        integrator: &impl ImplicitRates<
            NodalStatesSolidThermal,
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermalHistory,
            NodalRatesSolidThermalHistory,
        >,
        solver: impl RootFinding<
            NodalForcesSolidThermal,
            NodalStiffnessesSolidThermal,
            NodalStatesSolidThermal,
        >,
        time: &[Quantity<Time>],
        (mut initial_coordinates, mut initial_temperatures): (
            NodalCoordinates<3>,
            NodalTemperatures,
        ),
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError> {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let motion = Motion::new(
            conditions,
            initial_time,
            &mut initial_coordinates,
            &mut initial_temperatures,
            "implicit quasistatic thermoelastic-thermal transient",
            Stepping::Implicit,
        )?;
        let inverse = capacities
            .inverse(&motion.held_temperatures())
            .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse =
            solver_from_neighbors(&neighbors, &motion.constraint_at(initial_time), 4, false);
        let (times, states, rates) = integrator.integrate(
            |time, states: &NodalStatesSolidThermal| {
                let (coordinates, temperatures) = split_states(states);
                let at_rest = NodalVelocities::zero(coordinates.len());
                let mut internal = self
                    .nodal_forces(&coordinates, &at_rest, &temperatures)
                    .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, states, &mut internal)?;
                let unbalanced_heating = internal.iter().map(|force| force.1).collect();
                let temperature_rates = inverse.nodal_temperature_rates(&unbalanced_heating);
                Ok(join_rates(at_rest, temperature_rates))
            },
            |time, states: &NodalStatesSolidThermal, rates: &NodalRatesSolidThermal| {
                let (coordinates, temperatures) = split_states(states);
                let (velocities, temperature_rates) = split_rates(rates);
                let mut internal = self
                    .nodal_forces(&coordinates, &velocities, &temperatures)
                    .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, states, &mut internal)?;
                let capacitive_heating = capacities.capacitive_heating(&temperature_rates);
                Ok(internal
                    .into_iter()
                    .zip(capacitive_heating)
                    .map(|(internal, capacitive)| TensorTuple(internal.0, capacitive + internal.1))
                    .collect())
            },
            |_,
             states: &NodalStatesSolidThermal,
             rates: &NodalRatesSolidThermal,
             dt: Quantity<Time>| {
                let (coordinates, temperatures) = split_states(states);
                let (velocities, _) = split_rates(rates);
                let (mut tangent, heatings) = self
                    .nodal_stiffnesses(&coordinates, &velocities, &temperatures)
                    .map_err(|error| error.to_string())?;
                add_heatings(
                    &mut tangent,
                    &heatings,
                    Quantity::<Rate>::new(1.0 / dt.value()),
                );
                add_capacities(&mut tangent, capacities, 1.0 / dt.value());
                Ok(tangent)
            },
            solver,
            |time| motion.constraint_at(time),
            Some(sparse),
            time,
            join_states(&initial_coordinates, &initial_temperatures),
        )?;
        Ok((
            times,
            states.iter().map(|state| split_states(state).0).collect(),
            rates.iter().map(|rate| split_rates(rate).0).collect(),
            states.iter().map(|state| split_states(state).1).collect(),
            rates.iter().map(|rate| split_rates(rate).1).collect(),
        ))
    }
}
