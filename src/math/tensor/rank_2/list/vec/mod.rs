use crate::math::{TensorRank2List, TensorVector};

pub type TensorRank2ListVec<const D: usize, const I: usize, const J: usize, const W: usize> =
    TensorVector<TensorRank2List<D, I, J, W>>;
