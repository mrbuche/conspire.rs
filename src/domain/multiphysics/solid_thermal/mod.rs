pub(crate) mod thermoelastic_thermal_conduction;

use crate::{
    math::{
        Current, Quantity, TensorBlock2x2, TensorRank1, TensorRank2, TensorSparseVec2D,
        TensorTuple, TensorTupleVec,
    },
    units::{
        Force, ForcePerLength, ForcePerTemperature, Power, PowerPerLength, PowerPerTemperature,
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
