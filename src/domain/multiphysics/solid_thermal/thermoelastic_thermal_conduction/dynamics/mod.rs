pub mod implicit;
pub mod time_scale;

use self::time_scale::ThermoelasticThermalConductionTimeScaleElements;
use super::{motion::Motion, rates::ThermoelasticThermalConductionRateElements};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, SolidThermal},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, join_states,
        },
        solid::mass::{InverseMass, MassMatrix, NodalLumpedMasses},
        thermal::{
            NodalTemperatures,
            capacity::{
                HeatCapacityMatrix, InverseHeatCapacity, NodalLumpedHeatCapacities,
                NodalTemperatureRates,
            },
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Scalar, Tensor,
        integrate::{ExplicitDynamicsCoupled, IntegrationError, Spectrum, Times},
    },
    units::Time,
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
    NodalTemperaturesHistory,
    NodalTemperatureRatesHistory,
);

#[expect(clippy::too_many_arguments, clippy::type_complexity)]
fn held<'c, M, C, K>(
    masses: &M,
    capacities: &C,
    conditions: &'c K,
    initial_time: Quantity<Time>,
    coordinates: &mut NodalCoordinates<3>,
    temperatures: &mut NodalTemperatures,
    mut velocities: NodalVelocities<3>,
    integrator: &str,
    stepping: Stepping,
) -> Result<(M::Inverse, C::Inverse, NodalVelocities<3>, Motion<'c, K>), IntegrationError>
where
    M: MassMatrix<3>,
    C: HeatCapacityMatrix,
    K: ConstrainAt<SolidThermal<3>>,
{
    let motion = Motion::new(
        conditions,
        initial_time,
        coordinates,
        temperatures,
        integrator,
        stepping,
    )?;
    let masses = masses
        .inverse(&motion.held_coordinates())
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    let capacities = capacities
        .inverse(&motion.held_temperatures())
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    motion
        .fixed_coordinates()
        .iter()
        .for_each(|&index| velocities[index / 3][index % 3] = Default::default());
    Ok((masses, capacities, velocities, motion))
}

/// The motion and temperatures of a thermoelastic model that conducts heat.
pub trait ThermoelasticThermalConductionDynamics {
    #[expect(clippy::too_many_arguments)]
    fn nodal_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl InverseMass<3>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), ElementModelError>;
    /// Integrates the motion and temperatures of the model with an explicit coupled integrator.
    ///
    /// The masses and heat capacities may be lumped or consistent. Fixed degrees of freedom,
    /// whose indices are `4 * node + component` with the temperature as the last component, are
    /// held by giving them no velocity or rate of change, so that their initial values are the
    /// ones prescribed. Linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>;
    /// Integrates the motion and temperatures of the model like [`integrate`](Self::integrate),
    /// with a time step limited by the fastest time scales of the elements.
    ///
    /// Only lumped masses and heat capacities are accepted, since the estimates of the highest
    /// frequency and of the largest eigenvalue are those of the lumped ones, which consistent
    /// ones exceed. The estimates assemble the element stiffnesses, and are refreshed every
    /// `interval` evaluations of the bound, which for the integrators here is every step. The
    /// time step may use at most the fraction `safety` of the stability limit, which is the
    /// stricter of that of the motion along the imaginary axis and of the diffusion along the
    /// real axis, and a step above it is an error.
    #[expect(clippy::too_many_arguments)]
    fn integrate_bounded(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &NodalLumpedMasses,
        capacities: &NodalLumpedHeatCapacities,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermoelasticThermalConductionTimeScaleElements;
}

impl<B> ThermoelasticThermalConductionDynamics for Model<B, 3>
where
    B: ThermoelasticThermalConductionRateElements,
{
    fn nodal_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl InverseMass<3>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), ElementModelError> {
        let mut nodal_forces =
            self.nodal_forces(nodal_coordinates, nodal_velocities, nodal_temperatures)?;
        applied_loads
            .subtract_at(
                time,
                &join_states(nodal_coordinates, nodal_temperatures),
                &mut nodal_forces,
            )
            .map_err(|error| ElementModelError::upstream(error, "the applied loads"))?;
        let unbalanced_forces = nodal_forces.iter().map(|force| force.0.clone()).collect();
        let unbalanced_heating = nodal_forces.iter().map(|force| force.1).collect();
        Ok((
            masses.nodal_accelerations(&unbalanced_forces),
            capacities.nodal_temperature_rates(&unbalanced_heating),
        ))
    }
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError> {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (masses, capacities, initial_velocities, motion) = held(
            masses,
            capacities,
            conditions,
            initial_time,
            &mut initial_coordinates,
            &mut initial_temperatures,
            initial_velocities,
            "explicit thermoelastic-thermal dynamics",
            Stepping::Explicit,
        )?;
        let (times, mut coordinates, velocities, accelerations, mut temperatures, rates) =
            integrator.integrate_coupled(
                |time,
                 coordinates: &NodalCoordinates<3>,
                 velocities: &NodalVelocities<3>,
                 temperatures: &NodalTemperatures| {
                    let (mut coordinates, mut temperatures) =
                        (coordinates.clone(), temperatures.clone());
                    motion.apply(time, &mut coordinates, &mut temperatures);
                    self.nodal_rates(
                        &coordinates,
                        velocities,
                        &temperatures,
                        time,
                        applied_loads,
                        &masses,
                        &capacities,
                    )
                    .map_err(|error| error.to_string())
                },
                time,
                initial_coordinates,
                initial_velocities,
                initial_temperatures,
            )?;
        motion.record(&times, &mut coordinates, &mut temperatures);
        Ok((
            times,
            coordinates,
            velocities,
            accelerations,
            temperatures,
            rates,
        ))
    }
    fn integrate_bounded(
        &self,
        integrator: &impl ExplicitDynamicsCoupled<
            NodalCoordinates<3>,
            NodalTemperatures,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        mut initial_temperatures: NodalTemperatures,
        applied_loads: &impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal>,
        masses: &NodalLumpedMasses,
        capacities: &NodalLumpedHeatCapacities,
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermoelasticThermalConductionTimeScaleElements,
    {
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scales must be at least one."
                    .to_string(),
            ));
        }
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (masses, capacities, initial_velocities, motion) = held(
            masses,
            capacities,
            conditions,
            initial_time,
            &mut initial_coordinates,
            &mut initial_temperatures,
            initial_velocities,
            "explicit thermoelastic-thermal dynamics",
            Stepping::Explicit,
        )?;
        let mut evaluations = 0;
        let mut time_scales = (
            Time::seconds(Scalar::INFINITY),
            Time::seconds(Scalar::INFINITY),
        );
        let (times, mut coordinates, velocities, accelerations, mut temperatures, rates) =
            integrator.integrate_coupled_bounded(
                |time,
                 coordinates: &NodalCoordinates<3>,
                 velocities: &NodalVelocities<3>,
                 temperatures: &NodalTemperatures| {
                    let (mut coordinates, mut temperatures) =
                        (coordinates.clone(), temperatures.clone());
                    motion.apply(time, &mut coordinates, &mut temperatures);
                    self.nodal_rates(
                        &coordinates,
                        velocities,
                        &temperatures,
                        time,
                        applied_loads,
                        &masses,
                        &capacities,
                    )
                    .map_err(|error| error.to_string())
                },
                |time,
                 coordinates: &NodalCoordinates<3>,
                 _: &NodalVelocities<3>,
                 temperatures: &NodalTemperatures| {
                    if evaluations % interval == 0 {
                        let (mut coordinates, mut temperatures) =
                            (coordinates.clone(), temperatures.clone());
                        motion.apply(time, &mut coordinates, &mut temperatures);
                        time_scales = self
                            .fastest_time_scales(&self.coordinates, &coordinates, &temperatures)
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
            )?;
        motion.record(&times, &mut coordinates, &mut temperatures);
        Ok((
            times,
            coordinates,
            velocities,
            accelerations,
            temperatures,
            rates,
        ))
    }
}
