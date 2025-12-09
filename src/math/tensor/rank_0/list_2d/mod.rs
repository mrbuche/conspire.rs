use crate::math::{Tensor, TensorRank0, TensorRank0List, tensor::list::TensorList};
use std::ops::Mul;

pub type TensorRank0List2D<const N: usize> = TensorList<TensorRank0List<N>, N>;
