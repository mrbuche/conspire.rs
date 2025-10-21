#[cfg(test)]
mod test;

use crate::math::{TensorRank2, tensor::list::TensorList};

pub type TensorRank2List<const D: usize, const I: usize, const J: usize, const W: usize> =
    TensorList<W, TensorRank2<D, I, J>>;
