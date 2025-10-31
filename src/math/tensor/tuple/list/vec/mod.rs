use crate::math::{TensorTupleList, TensorVector};

pub type TensorTupleListVec<T1, T2, const N: usize> = TensorVector<TensorTupleList<T1, T2, N>>;
