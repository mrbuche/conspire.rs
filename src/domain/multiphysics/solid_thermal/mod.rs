pub(crate) mod thermoelastic_thermal_conduction;

use std::ops::Div;

use crate::{
    math::{
        Current, Quantity, TensorBlock2x2, TensorRank1, TensorRank2, TensorSparseVec2D,
        TensorTuple, TensorTupleVec,
    },
    units::{
        Force, ForcePerLength, ForcePerTemperature, Length, Power, PowerPerLength,
        PowerPerTemperature, Temperature,
    },
};

pub type NodalForceSolidThermal = TensorTuple<TensorRank1<3, Current, Force>, Quantity<Power>>;
pub type NodalForcesSolidThermal = TensorTupleVec<TensorRank1<3, Current, Force>, Quantity<Power>>;
pub type NodalStiffnessSolidThermal = TensorBlock2x2<
    TensorRank2<3, Current, Current, ForcePerLength>,
    TensorRank1<3, Current, ForcePerTemperature>,
    TensorRank1<3, Current, PowerPerLength>,
    Quantity<PowerPerTemperature>,
>;
pub type NodalStiffnessesSolidThermal = TensorSparseVec2D<NodalStiffnessSolidThermal>;
pub type NodalStatesSolidThermal =
    TensorTupleVec<TensorRank1<3, Current, Length>, Quantity<Temperature>>;

impl Div<NodalStiffnessesSolidThermal> for &NodalForcesSolidThermal {
    type Output = NodalStatesSolidThermal;
    fn div(self, _nodal_stiffnesses: NodalStiffnessesSolidThermal) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}
