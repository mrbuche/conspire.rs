use crate::{
    math::{Derivative, TensorTupleList},
    mechanics::{DeformationGradientPlastic, DeformationGradientRatePlastic},
};

pub type ViscoplasticStateVariables<const G: usize, Y> =
    TensorTupleList<DeformationGradientPlastic, Y, G>;
pub type ViscoplasticEvolution<const G: usize, Y> =
    TensorTupleList<DeformationGradientRatePlastic, Derivative<Y>, G>;
