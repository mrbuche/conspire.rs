//! Whether a run of explicit thermoelastic-thermal dynamics checks that its time step is stable.

use super::{Solution, time_scale::ThermoelasticThermalConductionTimeScaleElements};
use crate::{
    domain::{
        Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        lumping::Lumped,
        solid::mass::{LumpedMassElements, MassChoice},
        stability::{Checked, Unchecked},
        thermal::{
            NodalTemperatures,
            capacity::{CapacityChoice, LumpedHeatCapacityElements, NodalTemperatureRates},
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity,
        integrate::{ExplicitDynamicsCoupled, IntegrationError, Spectrum},
    },
    units::Time,
};

/// Whether a run of explicit thermoelastic-thermal dynamics checks that its time step is stable,
/// for a choice of masses and of heat capacities, which a policy that checks may not support.
pub trait Stability<M, C, K>
where
    C: MassChoice<M>,
    K: CapacityChoice<M>,
{
    /// Takes the steps, with the accelerations and the rates of the temperatures from `function`,
    /// and `held` gives the coordinates and temperatures as the motion that holds some of them
    /// has them.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        self,
        model: &M,
        held: impl Fn(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalTemperatures,
        ) -> (NodalCoordinates<3>, NodalTemperatures),
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
            &NodalTemperatures,
        ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), String>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError>;
}

impl<M, C, K> Stability<M, C, K> for Unchecked
where
    C: MassChoice<M>,
    K: CapacityChoice<M>,
{
    fn integrate(
        self,
        _: &M,
        _: impl Fn(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalTemperatures,
        ) -> (NodalCoordinates<3>, NodalTemperatures),
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
            &NodalTemperatures,
        ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), String>,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError> {
        integrator.integrate_coupled(
            function,
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
    }
}

impl<B> Stability<Model<B, 3>, Lumped, Lumped> for Checked
where
    B: LumpedMassElements<3> + LumpedHeatCapacityElements<3>,
    Model<B, 3>: ThermoelasticThermalConductionTimeScaleElements,
{
    fn integrate(
        self,
        model: &Model<B, 3>,
        held: impl Fn(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalTemperatures,
        ) -> (NodalCoordinates<3>, NodalTemperatures),
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
            &NodalTemperatures,
        ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), String>,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
    ) -> Result<Solution, IntegrationError> {
        let Self { safety, interval } = self;
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scales must be at least one."
                    .to_string(),
            ));
        }
        let mut evaluations = 0;
        let mut time_scales = (Time::seconds(f64::INFINITY), Time::seconds(f64::INFINITY));
        integrator.integrate_coupled_bounded(
            function,
            |time,
             coordinates: &NodalCoordinates<3>,
             _: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                if evaluations % interval == 0 {
                    let (coordinates, temperatures) = held(time, coordinates, temperatures);
                    time_scales = model
                        .fastest_time_scales(&model.coordinates, &coordinates, &temperatures)
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(Spectrum::Mixed(time_scales.1, time_scales.0))
            },
            safety,
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
    }
}
