use crate::math::{TensorRank2List, TensorVector};
use crate::units::Dimensionless;
/// A vector of lists of rank-2 tensors.
pub type TensorRank2ListVec<const D: usize, I, J, const W: usize, U = Dimensionless> =
    TensorVector<TensorRank2List<D, I, J, W, U>>;
