use crate::math::Dimensionless;
use crate::math::{TensorRank1Vec, tensor::vec::TensorVector};
/// A vector of vectors of rank-1 tensors.
pub type TensorRank1Vec2D<const D: usize, I, U = Dimensionless> =
    TensorVector<TensorRank1Vec<D, I, U>>;
