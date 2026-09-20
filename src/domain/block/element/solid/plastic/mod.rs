use crate::{
    math::{Quantity, TensorTupleList},
    mechanics::DeformationGradientPlastic,
};

/// The rate-independent plastic state at each of an element's `G` integration points:
/// the plastic deformation gradient and the equivalent plastic strain.
pub type PlasticStateVariables<const G: usize> =
    TensorTupleList<DeformationGradientPlastic, Quantity, G>;
