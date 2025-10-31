use crate::math::{TensorTuple, TensorVector};

pub type TensorTupleVec<T1, T2> = TensorVector<TensorTuple<T1, T2>>;
