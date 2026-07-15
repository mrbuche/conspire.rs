use crate::math::{TensorRank1Vec, tensor::vec::TensorVector};

/// A vector of vectors of rank-1 tensors.
pub type TensorRank1Vec2D<const D: usize, const I: usize> = TensorVector<TensorRank1Vec<D, I>>;
