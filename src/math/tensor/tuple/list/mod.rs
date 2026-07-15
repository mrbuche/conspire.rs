pub mod vec;
pub mod vec_2d;

use crate::math::{TensorList, TensorTuple};

/// A list of tensor tuples.
pub type TensorTupleList<T1, T2, const N: usize> = TensorList<TensorTuple<T1, T2>, N>;
