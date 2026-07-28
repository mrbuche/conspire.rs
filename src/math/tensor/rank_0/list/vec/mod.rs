use crate::math::{TensorRank0List, TensorVector};

/// A vector of lists of rank-0 tensors (scalars).
pub(crate) type TensorRank0ListVec<const W: usize> = TensorVector<TensorRank0List<W>>;
