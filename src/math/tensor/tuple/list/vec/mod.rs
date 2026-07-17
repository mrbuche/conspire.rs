use crate::math::{TensorTupleList, TensorVector};

/// A vector of lists of tensor tuples.
pub type TensorTupleListVec<T1, T2, const N: usize> = TensorVector<TensorTupleList<T1, T2, N>>;
