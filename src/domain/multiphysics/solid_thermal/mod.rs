pub(crate) mod thermoelastic_thermal_conduction;

use crate::{
    math::{Current, Quantity, TensorRank1, TensorTuple, TensorTupleVec},
    units::{Force, Power},
};

pub type NodalForceSolidThermal = TensorTuple<TensorRank1<3, Current, Force>, Quantity<Power>>;
pub type NodalForcesSolidThermal = TensorTupleVec<TensorRank1<3, Current, Force>, Quantity<Power>>;
