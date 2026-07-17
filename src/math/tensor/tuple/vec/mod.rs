use crate::math::{TensorTuple, TensorVector};

/// A vector of tensor tuples.
pub type TensorTupleVec<T1, T2> = TensorVector<TensorTuple<T1, T2>>;
