use crate::math::{TensorRank2ListVec, TensorVector};

pub type TensorRank2ListVec2D<const D: usize, const I: usize, const J: usize, const W: usize> =
    TensorVector<TensorRank2ListVec<D, I, J, W>>;
