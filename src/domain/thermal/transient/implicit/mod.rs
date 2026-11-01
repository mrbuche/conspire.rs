use super::{NodalTemperatureRatesHistory, NodalTemperaturesHistory, Solution, motion::Motion};
use crate::{
    domain::{
        Model, Stepping,
        boundary::{ConstrainAt, Field, LoadAt, LoadStiffnessAt},
        diffusion::DiffusionElements,
        thermal::{
            NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures,
            capacity::{
                CapacityChoice, InertialHeatCapacity, InverseHeatCapacity, NodalTemperatureRates,
            },
            conduction::ThermalConductionElements,
        },
    },
    math::{
        Quantity,
        integrate::{ImplicitRates, IntegrationError},
        optimize::RootFinding,
    },
    units::{Temperature, Time},
};

/// The temperatures of a model that conducts heat with implicit time integration.
pub trait ThermalConductionImplicitTransient
where
    Self: Sized,
{
    /// Integrates the temperatures of the model with an implicit integrator.
    ///
    /// The heat capacities are those of `capacity` ([`Lumped`](crate::domain::lumping::Lumped) or
    /// [`Consistent`](crate::domain::lumping::Consistent)) that the blocks of the model make.
    /// They are added to the sparse tangent.
    /// Fixed temperatures, whose indices are those of their nodes, are held at their initial
    /// values, and linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate<C>(
        &self,
        capacity: C,
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
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: CapacityChoice<Self>;
}

impl<B> ThermalConductionImplicitTransient for Model<B, 3>
where
    B: ThermalConductionElements,
{
    fn integrate<C>(
        &self,
        capacity: C,
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
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: CapacityChoice<Self>,
    {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let capacities = capacity.capacities(self);
        let (inverse, motion) = Motion::held(
            &capacities,
            conditions,
            initial_time,
            &mut initial_temperatures,
            "implicit thermal transient",
            Stepping::Implicit,
        )?;
        let sparse = self.sparse_solver(&motion.constraint_at(initial_time), 1, true);
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
