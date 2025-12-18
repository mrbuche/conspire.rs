#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank1List, tensor::list::TensorList};

pub type TensorRank1List2D<const D: usize, const I: usize, const M: usize, const N: usize> =
    TensorList<TensorRank1List<D, I, M>, N>;

impl<const D: usize, const I: usize, const M: usize, const N: usize>
    From<[[[TensorRank0; D]; M]; N]> for TensorRank1List2D<D, I, M, N>
{
    fn from(array: [[[TensorRank0; D]; M]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
