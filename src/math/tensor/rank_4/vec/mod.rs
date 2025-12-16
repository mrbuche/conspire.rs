use crate::math::{TensorRank4, tensor::vec::TensorVector};

pub type TensorRank4Vec<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
> = TensorVector<TensorRank4<D, I, J, K, L>>;
