pub(crate) mod thermoelastic_thermal_conduction;

use crate::{
    domain::{
        NodalCoordinates, NodalVelocities,
        thermal::{NodalTemperatures, capacity::NodalTemperatureRates},
    },
    math::{
        Current, Quantity, Tensor, TensorBlock2x2, TensorRank1, TensorRank2, TensorSparseVec2D,
        TensorTuple, TensorTupleVec, TensorVector,
    },
    units::{
        Force, ForcePerLength, ForcePerTemperature, Length, Power, PowerPerLength,
        PowerPerTemperature, Temperature, TemperatureRate, Velocity,
    },
};
use std::ops::Div;

pub type NodalForceSolidThermal = TensorTuple<TensorRank1<3, Current, Force>, Quantity<Power>>;
pub type NodalForcesSolidThermal = TensorTupleVec<TensorRank1<3, Current, Force>, Quantity<Power>>;
pub type NodalStiffnessSolidThermal = TensorBlock2x2<
    TensorRank2<3, Current, Current, ForcePerLength>,
    TensorRank1<3, Current, ForcePerTemperature>,
    TensorRank1<3, Current, PowerPerLength>,
    Quantity<PowerPerTemperature>,
>;
pub type NodalStiffnessesSolidThermal = TensorSparseVec2D<NodalStiffnessSolidThermal>;
pub type NodalHeatingSolidThermal = TensorRank1<3, Current, Force>;
pub type NodalHeatingsSolidThermal = TensorSparseVec2D<NodalHeatingSolidThermal>;
pub type NodalStatesSolidThermal =
    TensorTupleVec<TensorRank1<3, Current, Length>, Quantity<Temperature>>;
pub type NodalStatesSolidThermalHistory = TensorVector<NodalStatesSolidThermal>;
pub type NodalRatesSolidThermal =
    TensorTupleVec<TensorRank1<3, Current, Velocity>, Quantity<TemperatureRate>>;
pub type NodalRatesSolidThermalHistory = TensorVector<NodalRatesSolidThermal>;

pub(crate) fn split_states(
    states: &NodalStatesSolidThermal,
) -> (NodalCoordinates<3>, NodalTemperatures) {
    (
        states.iter().map(|state| state.0.clone()).collect(),
        states.iter().map(|state| state.1).collect(),
    )
}

pub(crate) fn join_states(
    coordinates: &NodalCoordinates<3>,
    temperatures: &NodalTemperatures,
) -> NodalStatesSolidThermal {
    coordinates
        .iter()
        .zip(temperatures.iter())
        .map(|(coordinate, &temperature)| TensorTuple(coordinate.clone(), temperature))
        .collect()
}

pub(crate) fn split_rates(
    rates: &NodalRatesSolidThermal,
) -> (NodalVelocities<3>, NodalTemperatureRates) {
    (
        rates.iter().map(|rate| rate.0.clone()).collect(),
        rates.iter().map(|rate| rate.1).collect(),
    )
}

pub(crate) fn join_rates(
    velocities: NodalVelocities<3>,
    temperature_rates: NodalTemperatureRates,
) -> NodalRatesSolidThermal {
    velocities
        .into_iter()
        .zip(temperature_rates)
        .map(|(velocity, rate)| TensorTuple(velocity, rate))
        .collect()
}

impl Div<NodalStiffnessesSolidThermal> for &NodalForcesSolidThermal {
    type Output = NodalStatesSolidThermal;
    fn div(self, _nodal_stiffnesses: NodalStiffnessesSolidThermal) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}
