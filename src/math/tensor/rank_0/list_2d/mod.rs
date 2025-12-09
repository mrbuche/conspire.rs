use crate::math::{TensorList, TensorRank0List};

pub type TensorRank0List2D<const N: usize> = TensorList<TensorRank0List<N>, N>;
