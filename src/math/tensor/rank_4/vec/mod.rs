use crate::math::{TensorRank4, tensor::vec::TensorVector};
use crate::units::Dimensionless;
/// A vector of rank-4 tensors.
pub type TensorRank4Vec<const D: usize, I, J, K, L, U = Dimensionless> =
    TensorVector<TensorRank4<D, I, J, K, L, U>>;
