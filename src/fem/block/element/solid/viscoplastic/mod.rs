use crate::{
    math::{Scalar, TensorTupleList},
    mechanics::DeformationGradientPlastic,
};

pub type ViscoplasticStateVariables<const G: usize> =
    TensorTupleList<DeformationGradientPlastic, Scalar, G>;
