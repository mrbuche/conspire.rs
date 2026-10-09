use crate::{
    domain::{
        ElementModelError, Model,
        factor::fixed_indices,
        thermal::{
            NodalForcesThermal, NodalTemperatures,
            capacity::{
                HeatCapacityMatrix, InverseHeatCapacity, NodalLumpedHeatCapacities,
                NodalTemperatureRates,
            },
            conduction::ThermalConductionElements,
            time_scale::ThermalTimeScaleElements,
        },
    },
    math::{
        Quantity, Scalar, TensorVector,
        integrate::{Explicit, IntegrationError, Spectrum, Times},
        optimize::EqualityConstraint,
    },
    units::Time,
};

pub type NodalTemperaturesHistory = TensorVector<NodalTemperatures>;
pub type NodalTemperatureRatesHistory = TensorVector<NodalTemperatureRates>;

type Solution = (
    Times,
    NodalTemperaturesHistory,
    NodalTemperatureRatesHistory,
);

fn held<M>(
    capacities: &M,
    equality_constraint: EqualityConstraint,
) -> Result<M::Inverse, IntegrationError>
where
    M: HeatCapacityMatrix,
{
    let fixed = fixed_indices(equality_constraint, "explicit thermal dynamics")?;
    capacities
        .inverse(&fixed)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))
}

/// The temperatures of a model that conducts heat.
pub trait ThermalConductionDynamics {
    fn nodal_temperature_rates(
        &self,
        nodal_temperatures: &NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<NodalTemperatureRates, ElementModelError>;
    /// Integrates the temperatures of the model with an explicit integrator.
    ///
    /// The heat capacities may be lumped or consistent. Fixed temperatures, whose indices are
    /// those of their nodes, are held by giving them no rate of change, so that their initial
    /// values are the ones prescribed. Linear constraints are not supported.
    fn integrate(
        &self,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>;
    /// Integrates the temperatures of the model like [`integrate`](Self::integrate), with a
    /// time step limited by the fastest time scale of the elements.
    ///
    /// Only lumped heat capacities are accepted, since the estimate of the largest eigenvalue
    /// is that of the lumped heat capacities, which a consistent one exceeds. The estimate
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
        external_heating: &NodalForcesThermal,
        capacities: &NodalLumpedHeatCapacities,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermalTimeScaleElements;
}

impl<B, const D: usize> ThermalConductionDynamics for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn nodal_temperature_rates(
        &self,
        nodal_temperatures: &NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &impl InverseHeatCapacity,
    ) -> Result<NodalTemperatureRates, ElementModelError> {
        Ok(capacities
            .nodal_temperature_rates(external_heating, &self.nodal_forces(nodal_temperatures)?))
    }
    fn integrate(
        &self,
        integrator: &impl Explicit<
            NodalTemperatures,
            NodalTemperaturesHistory,
            NodalTemperatureRatesHistory,
        >,
        time: &[Quantity<Time>],
        initial_temperatures: NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &impl HeatCapacityMatrix,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError> {
        let capacities = held(capacities, equality_constraint)?;
        integrator.integrate(
            |_, temperatures: &NodalTemperatures| {
                self.nodal_temperature_rates(temperatures, external_heating, &capacities)
                    .map_err(|error| error.to_string())
            },
            time,
            initial_temperatures,
        )
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
        initial_temperatures: NodalTemperatures,
        external_heating: &NodalForcesThermal,
        capacities: &NodalLumpedHeatCapacities,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>
    where
        Self: ThermalTimeScaleElements,
    {
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let capacities = held(capacities, equality_constraint)?;
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(Scalar::INFINITY);
        integrator.integrate_bounded(
            |_, temperatures: &NodalTemperatures| {
                self.nodal_temperature_rates(temperatures, external_heating, &capacities)
                    .map_err(|error| error.to_string())
            },
            |_, temperatures: &NodalTemperatures| {
                if evaluations % interval == 0 {
                    time_scale = self
                        .fastest_diffusive_time_scale(temperatures)
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
