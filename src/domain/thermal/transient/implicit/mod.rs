use super::{NodalTemperatureRatesHistory, NodalTemperaturesHistory, Solution, held};
use crate::{
    domain::{
        ElementModel, Model,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
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
        optimize::{EqualityConstraints, RootFinding},
    },
    units::Time,
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
        external_heating: &NodalForcesThermal,
        capacities: &impl InertialHeatCapacity,
        equality_constraint: EqualityConstraints,
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
        initial_temperatures: NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &impl InertialHeatCapacity,
        equality_constraint: EqualityConstraints,
    ) -> Result<Solution, IntegrationError> {
        let inverse = held(
            capacities,
            equality_constraint.clone(),
            "implicit thermal transient",
        )?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, 1, true);
        integrator.integrate(
            |_, temperatures: &NodalTemperatures| {
                Ok(inverse.nodal_temperature_rates(
                    external_heating,
                    &self
                        .nodal_forces(temperatures)
                        .map_err(|error| error.to_string())?,
                ))
            },
            |_, temperatures: &NodalTemperatures, rates: &NodalTemperatureRates| {
                Ok(capacities.capacitive_heating(rates)
                    + self
                        .nodal_forces(temperatures)
                        .map_err(|error| error.to_string())?
                    - external_heating)
            },
            |_, temperatures: &NodalTemperatures, _: &NodalTemperatureRates, dt| {
                let mut tangent = self
                    .nodal_stiffnesses(temperatures)
                    .map_err(|error| error.to_string())?;
                capacities.add_to_stiffnesses(1.0 / dt.value(), &mut tangent);
                Ok(tangent)
            },
            solver,
            equality_constraint,
            Some(sparse),
            time,
            initial_temperatures,
        )
    }
}
