//! Whether a run of explicit thermal integration checks that its time step is stable.

#[cfg(all(test, feature = "cbm"))]
mod test;

use super::{NodalTemperatureRatesHistory, NodalTemperaturesHistory, Solution};
use crate::{
    domain::{
        Model,
        lumping::Lumped,
        stability::{Checked, Unchecked},
        thermal::{
            NodalTemperatures,
            capacity::{CapacityChoice, LumpedHeatCapacityElements, NodalTemperatureRates},
            time_scale::ThermalTimeScaleElements,
        },
    },
    math::{
        Quantity,
        integrate::{Explicit, IntegrationError, Spectrum},
    },
    units::Time,
};

/// Whether a run of explicit thermal integration checks that its time step is stable, for a
/// choice of heat capacities, which a policy that checks may not support.
pub trait Stability<M, C>
where
    C: CapacityChoice<M>,
{
    /// Takes the steps, with the rates of the temperatures from `function`, and `held` gives the
    /// temperatures as the motion that holds some of them has them.
    fn integrate(
        self,
        model: &M,
        held: impl Fn(Quantity<Time>, &NodalTemperatures) -> NodalTemperatures,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalTemperatures,
        ) -> Result<NodalTemperatureRates, String>,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError>;
}

impl<M, C> Stability<M, C> for Unchecked
where
    C: CapacityChoice<M>,
{
    fn integrate(
        self,
        _: &M,
        _: impl Fn(Quantity<Time>, &NodalTemperatures) -> NodalTemperatures,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalTemperatures,
        ) -> Result<NodalTemperatureRates, String>,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError> {
        integrator.integrate(function, time, initial_temperatures)
    }
}

impl<B> Stability<Model<B, 3>, Lumped> for Checked
where
    B: LumpedHeatCapacityElements<3>,
    Model<B, 3>: ThermalTimeScaleElements<3>,
{
    fn integrate(
        self,
        model: &Model<B, 3>,
        held: impl Fn(Quantity<Time>, &NodalTemperatures) -> NodalTemperatures,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalTemperatures,
        ) -> Result<NodalTemperatureRates, String>,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError> {
        let Self { safety, interval } = self;
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(f64::INFINITY);
        integrator.integrate_bounded(
            function,
            |time, temperatures: &NodalTemperatures| {
                if evaluations % interval == 0 {
                    time_scale = model
                        .fastest_diffusive_time_scale(&model.coordinates, &held(time, temperatures))
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(Spectrum::Real(time_scale))
            },
            safety,
            time,
            initial_temperatures,
        )
    }
}
