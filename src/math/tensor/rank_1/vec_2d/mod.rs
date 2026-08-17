use crate::math::{TensorRank1Vec, tensor::vec::TensorVector};
use crate::units::Dimensionless;
/// A vector of vectors of rank-1 tensors.
pub type TensorRank1Vec2D<const D: usize, I, U = Dimensionless> =
    TensorVector<TensorRank1Vec<D, I, U>>;
