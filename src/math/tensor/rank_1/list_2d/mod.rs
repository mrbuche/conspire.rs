#[cfg(test)]
mod test;

use crate::math::{TensorRank1List, tensor::list::TensorList};

pub type TensorRank1List2D<const D: usize, const I: usize, const W: usize, const X: usize> =
    TensorList<TensorRank1List<D, I, W>, X>;
