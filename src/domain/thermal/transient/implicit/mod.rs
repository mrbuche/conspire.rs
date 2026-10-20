use super::{NodalTemperatureRatesHistory, NodalTemperaturesHistory, Solution, motion::Motion};
use crate::{
    domain::{
        ElementModel, Model, Stepping,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{ConstrainAt, Field, LoadAt, LoadStiffnessAt},
        diffusion::DiffusionElements,
        thermal::{
            NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures,
            capacity::{InertialHeatCapacity, InverseHeatCapacity, NodalTemperatureRates},
            conduction::ThermalConductionElements,
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{ImplicitRates, IntegrationError},
        optimize::RootFinding,
    },
    units::{Temperature, Time},
};

/// The temperatures of a model that conducts heat with implicit time integration.
pub trait ThermalConductionImplicitTransient {
    /// Integrates the temperatures of the model with an implicit integrator.
    ///
    /// The heat capacities may be lumped or consistent, and are added to the sparse tangent.
    /// Fixed temperatures, whose indices are those of their nodes, are held at their initial
    /// values, and linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ImplicitRates<
            NodalTemperatures,
            NodalForcesThermal,
            NodalStiffnessesThermal,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        solver: impl RootFinding<NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures>,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
        applied_loads: &(
             impl LoadAt<NodalTemperatures, NodalForcesThermal>
             + LoadStiffnessAt<NodalTemperatures, NodalStiffnessesThermal>
         ),
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ThermalConductionImplicitTransient for Model<B, 3>
where
    B: ThermalConductionElements,
{
    fn integrate(
        &self,
        integrator: &impl ImplicitRates<
            NodalTemperatures,
            NodalForcesThermal,
            NodalStiffnessesThermal,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        solver: impl RootFinding<NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures>,
        time: &[Quantity<Time>],
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &(
             impl LoadAt<NodalTemperatures, NodalForcesThermal>
             + LoadStiffnessAt<NodalTemperatures, NodalStiffnessesThermal>
         ),
        capacities: &impl InertialHeatCapacity,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError> {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (inverse, motion) = Motion::held(
            capacities,
            conditions,
            initial_time,
            &mut initial_temperatures,
            "implicit thermal transient",
            Stepping::Implicit,
        )?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse =
            solver_from_neighbors(&neighbors, &motion.constraint_at(initial_time), 1, true);
        integrator.integrate(
            |time, temperatures: &NodalTemperatures| {
                let mut unbalanced_heating = self
                    .nodal_forces(temperatures)
                    .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, temperatures, &mut unbalanced_heating)?;
                Ok(inverse.nodal_temperature_rates(&unbalanced_heating))
            },
            |time, temperatures: &NodalTemperatures, rates: &NodalTemperatureRates| {
                let mut residual = capacities.capacitive_heating(rates)
                    + self
                        .nodal_forces(temperatures)
                        .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, temperatures, &mut residual)?;
                Ok(residual)
            },
            |time, temperatures: &NodalTemperatures, _: &NodalTemperatureRates, dt| {
                let mut tangent = self
                    .nodal_stiffnesses(temperatures)
                    .map_err(|error| error.to_string())?;
                capacities.add_to_stiffnesses(1.0 / dt.value(), &mut tangent);
                applied_loads.add_stiffness_at(time, temperatures, &mut tangent)?;
                Ok(tangent)
            },
            solver,
            |time| motion.constraint_at(time),
            Some(sparse),
            time,
            initial_temperatures,
        )
    }
}
