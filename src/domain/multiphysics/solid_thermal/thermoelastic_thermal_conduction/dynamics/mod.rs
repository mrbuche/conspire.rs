pub mod implicit;
pub mod time_scale;

use self::time_scale::ThermoelasticThermalConductionTimeScaleElements;
use super::rates::ThermoelasticThermalConductionRateElements;
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        factor::fixed_indices,
        solid::{
            NodalForcesSolid,
            mass::{InverseMass, MassMatrix, NodalLumpedMasses},
        },
        thermal::{
            NodalForcesThermal, NodalTemperatures,
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
        optimize::EqualityConstraints,
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

fn held<M, C>(
    masses: &M,
    capacities: &C,
    equality_constraint: EqualityConstraints,
    mut velocities: NodalVelocities<3>,
) -> Result<(M::Inverse, C::Inverse, NodalVelocities<3>), IntegrationError>
where
    M: MassMatrix<3>,
    C: HeatCapacityMatrix,
{
    let fixed = fixed_indices(
        equality_constraint,
        "explicit thermoelastic-thermal dynamics",
    )?;
    let fixed_coordinates = fixed
        .iter()
        .filter(|&&index| index % 4 < 3)
        .map(|&index| 3 * (index / 4) + index % 4)
        .collect::<Vec<_>>();
    let fixed_temperatures = fixed
        .iter()
        .filter(|&&index| index % 4 == 3)
        .map(|&index| index / 4)
        .collect::<Vec<_>>();
    let masses = masses
        .inverse(&fixed_coordinates)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    let capacities = capacities
        .inverse(&fixed_temperatures)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    fixed_coordinates
        .iter()
        .for_each(|&index| velocities[index / 3][index % 3] = Default::default());
    Ok((masses, capacities, velocities))
}

/// The motion and temperatures of a thermoelastic model that conducts heat.
pub trait ThermoelasticThermalConductionDynamics {
    #[expect(clippy::too_many_arguments)]
    fn nodal_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
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
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraints,
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
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &NodalLumpedMasses,
        capacities: &NodalLumpedHeatCapacities,
        equality_constraint: EqualityConstraints,
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
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl InverseMass<3>,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<(NodalAccelerations<3>, NodalTemperatureRates), ElementModelError> {
        let nodal_forces =
            self.nodal_forces(nodal_coordinates, nodal_velocities, nodal_temperatures)?;
        let internal_forces = nodal_forces.iter().map(|force| force.0.clone()).collect();
        let internal_heating = nodal_forces.iter().map(|force| force.1).collect();
        Ok((
            masses.nodal_accelerations(external_forces, &internal_forces),
            capacities.nodal_temperature_rates(external_heating, &internal_heating),
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
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &impl MassMatrix<3>,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraints,
    ) -> Result<Solution, IntegrationError> {
        let (masses, capacities, initial_velocities) =
            held(masses, capacities, equality_constraint, initial_velocities)?;
        integrator.integrate_coupled(
            |_,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                self.nodal_rates(
                    coordinates,
                    velocities,
                    temperatures,
                    external_forces,
                    external_heating,
                    &masses,
                    &capacities,
                )
                .map_err(|error| error.to_string())
            },
            time,
            initial_coordinates,
            initial_velocities,
            initial_temperatures,
        )
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
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        initial_temperatures: NodalTemperatures,
        external_forces: &NodalForcesSolid<3>,
        external_heating: &NodalForcesThermal,
        masses: &NodalLumpedMasses,
        capacities: &NodalLumpedHeatCapacities,
        equality_constraint: EqualityConstraints,
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
        let (masses, capacities, initial_velocities) =
            held(masses, capacities, equality_constraint, initial_velocities)?;
        let mut evaluations = 0;
        let mut time_scales = (
            Time::seconds(Scalar::INFINITY),
            Time::seconds(Scalar::INFINITY),
        );
        integrator.integrate_coupled_bounded(
            |_,
             coordinates: &NodalCoordinates<3>,
             velocities: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                self.nodal_rates(
                    coordinates,
                    velocities,
                    temperatures,
                    external_forces,
                    external_heating,
                    &masses,
                    &capacities,
                )
                .map_err(|error| error.to_string())
            },
            |_,
             coordinates: &NodalCoordinates<3>,
             _: &NodalVelocities<3>,
             temperatures: &NodalTemperatures| {
                if evaluations % interval == 0 {
                    time_scales = self
                        .fastest_time_scales(&self.coordinates, coordinates, temperatures)
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
