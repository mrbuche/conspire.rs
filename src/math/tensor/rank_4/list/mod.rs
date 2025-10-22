#[cfg(test)]
mod test;

use crate::math::{TensorRank4, tensor::list::TensorList};

pub type TensorRank4List<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const W: usize,
> = TensorList<TensorRank4<D, I, J, K, L>, W>;
