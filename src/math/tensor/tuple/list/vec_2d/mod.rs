use crate::math::{TensorTupleListVec, TensorVector};

pub type TensorTupleListVec2D<T1, T2, const N: usize> = TensorVector<TensorTupleListVec<T1, T2, N>>;
