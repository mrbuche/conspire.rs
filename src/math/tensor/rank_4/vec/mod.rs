use crate::math::{TensorRank4, tensor::vec::TensorVector};
/// A vector of rank-4 tensors.
pub type TensorRank4Vec<const D: usize, I, J, K, L> = TensorVector<TensorRank4<D, I, J, K, L>>;
