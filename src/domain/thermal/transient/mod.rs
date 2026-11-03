#[cfg(test)]
pub(crate) mod test;

mod implicit;
mod motion;
mod stability;

#[cfg_attr(not(feature = "fem"), allow(unused_imports))]
pub use crate::domain::stability::{Checked, Unchecked};
pub use implicit::ThermalConductionImplicitTransient;
pub use stability::Stability;

use crate::{
    domain::{
        ElementModelError, Model, Stepping,
        boundary::{ConstrainAt, Field, LoadAt},
        diffusion::DiffusionElements,
        thermal::{
            NodalForcesThermal, NodalTemperatures,
            capacity::{CapacityChoice, InverseHeatCapacity, NodalTemperatureRates},
            conduction::ThermalConductionElements,
        },
    },
    math::{
        Quantity, TensorVector,
        integrate::{Explicit, IntegrationError, Times},
    },
    units::{Temperature, Time},
};
use motion::Motion;

pub type NodalTemperaturesHistory = TensorVector<NodalTemperatures>;
pub type NodalTemperatureRatesHistory = TensorVector<NodalTemperatureRates>;

type Solution = (
    Times,
    NodalTemperaturesHistory,
    NodalTemperatureRatesHistory,
);

/// The temperatures of a model that conducts heat.
pub trait ThermalConductionTransient
where
    Self: Sized,
{
    fn nodal_temperature_rates(
        &self,
        nodal_temperatures: &NodalTemperatures,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<NodalTemperatureRates, ElementModelError>;
    /// Integrates the temperatures of the model with an explicit integrator.
    ///
    /// The heat capacities are those of `capacity` ([`Lumped`](crate::domain::lumping::Lumped) or
    /// [`Consistent`](crate::domain::lumping::Consistent)) that the blocks of the model make, and
    /// `stability` ([`Unchecked`] or [`Checked`]) says whether the time step is checked against
    /// the stability limit, which a policy may only do for some choices of heat capacities.
    /// Fixed temperatures, whose indices are those of their nodes, are held by giving them no
    /// rate of change, so that their initial values are the ones prescribed. Linear constraints
    /// are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate<C>(
        &self,
        stability: impl Stability<Self, C>,
        capacity: C,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: CapacityChoice<Self>;
}

impl<B> ThermalConductionTransient for Model<B, 3>
where
    B: ThermalConductionElements,
{
    fn nodal_temperature_rates(
        &self,
        nodal_temperatures: &NodalTemperatures,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<NodalTemperatureRates, ElementModelError> {
        let mut unbalanced_heating = self.nodal_forces(nodal_temperatures)?;
        applied_loads
            .subtract_at(time, nodal_temperatures, &mut unbalanced_heating)
            .map_err(|error| ElementModelError::upstream(error, "the applied loads"))?;
        Ok(capacities.nodal_temperature_rates(&unbalanced_heating))
    }
    fn integrate<C>(
        &self,
        stability: impl Stability<Self, C>,
        capacity: C,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: CapacityChoice<Self>,
    {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let capacities = capacity.capacities(self);
        let (capacities, motion) = Motion::held(
            &capacities,
            conditions,
            initial_time,
            &mut initial_temperatures,
            "explicit thermal transient",
            Stepping::Explicit,
        )?;
        let (times, mut temperatures, rates) = stability.integrate(
            self,
            |time, temperatures| {
                let mut temperatures = temperatures.clone();
                motion.apply(time, &mut temperatures);
                temperatures
            },
            integrator,
            |time, temperatures: &NodalTemperatures| {
                let mut temperatures = temperatures.clone();
                motion.apply(time, &mut temperatures);
                self.nodal_temperature_rates(&temperatures, time, applied_loads, &capacities)
                    .map_err(|error| error.to_string())
            },
            time,
            initial_temperatures,
        )?;
        motion.record(&times, &mut temperatures);
        Ok((times, temperatures, rates))
    }
}
