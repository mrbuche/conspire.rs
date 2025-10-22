use crate::math::{TensorRank1Vec, tensor::vec::TensorVector};

pub type TensorRank1Vec2D<const D: usize, const I: usize> = TensorVector<TensorRank1Vec<D, I>>;
