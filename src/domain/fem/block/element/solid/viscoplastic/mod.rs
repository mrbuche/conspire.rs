use crate::{math::TensorTupleList, mechanics::DeformationGradientPlastic};

pub type ViscoplasticStateVariables<const G: usize, Y> =
    TensorTupleList<DeformationGradientPlastic, Y, G>;
