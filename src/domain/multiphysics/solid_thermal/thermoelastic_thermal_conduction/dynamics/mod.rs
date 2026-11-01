pub mod implicit;
mod stability;
pub mod time_scale;

#[cfg_attr(not(feature = "fem"), allow(unused_imports))]
pub use crate::domain::stability::{Checked, Unchecked};
pub use stability::Stability;

use super::{motion::Motion, rates::ThermoelasticThermalConductionRateElements};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, SolidThermal},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, join_states,
        },
        solid::mass::{InverseMass, MassChoice, MassMatrix},
        thermal::{
            NodalTemperatures,
            capacity::{
                CapacityChoice, HeatCapacityMatrix, InverseHeatCapacity, NodalTemperatureRates,
            },
            transient::{NodalTemperatureRatesHistory, NodalTemperaturesHistory},
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{ExplicitDynamicsCoupled, IntegrationError, Times},
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
pub trait ThermoelasticThermalConductionDynamics
where
    Self: Sized,
{
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
    /// The masses are those of `mass` ([`Lumped`](crate::domain::lumping::Lumped) or
    /// [`Consistent`](crate::domain::lumping::Consistent)) that the blocks of the model make,
    /// and the heat capacities those of `capacity`, in the same way.
    /// `stability` ([`Unchecked`] or [`Checked`]) says whether the time step is checked against
    /// the stability limit, which a policy may only do for some choices of masses and heat
    /// capacities. Fixed degrees of freedom, whose indices are `4 * node + component` with the temperature as the last component, are
    /// held by giving them no velocity or rate of change, so that their initial values are the
    /// ones prescribed. Linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate<C, K>(
        &self,
        stability: impl Stability<Self, C, K>,
        mass: C,
        capacity: K,
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
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>,
        K: CapacityChoice<Self>;
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
    fn integrate<C, K>(
        &self,
        stability: impl Stability<Self, C, K>,
        mass: C,
        capacity: K,
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
        conditions: &impl ConstrainAt<SolidThermal<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>,
        K: CapacityChoice<Self>,
    {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let masses = mass.masses(self);
        let capacities = capacity.capacities(self);
        let (masses, capacities, initial_velocities, motion) = held(
            &masses,
            &capacities,
            conditions,
            initial_time,
            &mut initial_coordinates,
            &mut initial_temperatures,
            initial_velocities,
            "explicit thermoelastic-thermal dynamics",
            Stepping::Explicit,
        )?;
        let (times, mut coordinates, velocities, accelerations, mut temperatures, rates) =
            stability.integrate(
                self,
                |time, coordinates, temperatures| {
                    let (mut coordinates, mut temperatures) =
                        (coordinates.clone(), temperatures.clone());
                    motion.apply(time, &mut coordinates, &mut temperatures);
                    (coordinates, temperatures)
                },
                integrator,
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
                (initial_coordinates, initial_velocities),
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
