#[cfg(test)]
pub(crate) mod test;

mod implicit;
mod motion;

pub use implicit::ThermalConductionImplicitTransient;

use crate::{
    domain::{
        ElementModelError, Model, Stepping,
        boundary::{ConstrainAt, Field, LoadAt},
        diffusion::DiffusionElements,
        lumping::Lumped,
        thermal::{
            NodalForcesThermal, NodalTemperatures,
            capacity::{CapacityChoice, InverseHeatCapacity, NodalTemperatureRates},
            conduction::ThermalConductionElements,
            time_scale::ThermalTimeScaleElements,
        },
    },
    math::{
        Quantity, Scalar, TensorVector,
        integrate::{Explicit, IntegrationError, Spectrum, Times},
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
    /// [`Consistent`](crate::domain::lumping::Consistent)) that the blocks of the model make.
    /// Fixed temperatures, whose indices are those of their nodes, are held by giving them no
    /// rate of change, so that their initial values are the ones prescribed. Linear constraints
    /// are not supported.
    fn integrate<C>(
        &self,
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
    /// Integrates the temperatures of the model like [`integrate`](Self::integrate), with a
    /// time step limited by the fastest time scale of the elements.
    ///
    /// The heat capacities are the lumped ones of the model, since the estimate of the largest
    /// eigenvalue is that of the lumped heat capacities, which a consistent one exceeds. The estimate
    /// assembles the element conduction tangents, and is refreshed every `interval`
    /// evaluations of the bound, which for the integrators here is every step. The time step
    /// may use at most the fraction `safety` of the stability limit, and a step above it is an
    /// error.
    #[expect(clippy::too_many_arguments)]
    fn integrate_bounded(
        &self,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermalTimeScaleElements<3>,
        Lumped: CapacityChoice<Self>;
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
        let (times, mut temperatures, rates) = integrator.integrate(
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
    fn integrate_bounded(
        &self,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalTemperatures, NodalForcesThermal>,
        conditions: &impl ConstrainAt<Field<Temperature>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermalTimeScaleElements<3>,
        Lumped: CapacityChoice<Self>,
    {
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let capacities = Lumped.capacities(self);
        let (capacities, motion) = Motion::held(
            &capacities,
            conditions,
            initial_time,
            &mut initial_temperatures,
            "explicit thermal transient",
            Stepping::Explicit,
        )?;
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(Scalar::INFINITY);
        let (times, mut temperatures, rates) = integrator.integrate_bounded(
            |time, temperatures: &NodalTemperatures| {
                let mut temperatures = temperatures.clone();
                motion.apply(time, &mut temperatures);
                self.nodal_temperature_rates(&temperatures, time, applied_loads, &capacities)
                    .map_err(|error| error.to_string())
            },
            |time, temperatures: &NodalTemperatures| {
                if evaluations % interval == 0 {
                    let mut temperatures = temperatures.clone();
                    motion.apply(time, &mut temperatures);
                    time_scale = self
                        .fastest_diffusive_time_scale(&self.coordinates, &temperatures)
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(Spectrum::Real(time_scale))
            },
            safety,
            time,
            initial_temperatures,
        )?;
        motion.record(&times, &mut temperatures);
        Ok((times, temperatures, rates))
    }
}
