use crate::{
    math::{Quantity, TensorTupleListVec},
    mechanics::DeformationGradientPlastic,
};

/// The rate-independent plastic state at every integration point of every element in a block.
pub type PlasticStateVariablesField<const G: usize> =
    TensorTupleListVec<DeformationGradientPlastic, Quantity, G>;
