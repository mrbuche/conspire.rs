#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank1List, tensor::list::TensorList};
use crate::units::Dimensionless;

/// A list of lists of rank-1 tensors.
pub type TensorRank1List2D<const D: usize, I, const M: usize, const N: usize, U = Dimensionless> =
    TensorList<TensorRank1List<D, I, M, U>, N>;

impl<const D: usize, I, const M: usize, const N: usize, U> From<[[[TensorRank0; D]; M]; N]>
    for TensorRank1List2D<D, I, M, N, U>
{
    fn from(array: [[[TensorRank0; D]; M]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
