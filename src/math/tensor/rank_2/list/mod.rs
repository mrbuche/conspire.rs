use crate::math::Dimensionless;
pub(crate) mod vec;

#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank2, tensor::list::TensorList};

/// A list of rank-2 tensors.
pub type TensorRank2List<const D: usize, I, J, const N: usize, U = Dimensionless> =
    TensorList<TensorRank2<D, I, J, U>, N>;

impl<const D: usize, I, J, const N: usize, U> From<[[[TensorRank0; D]; D]; N]>
    for TensorRank2List<D, I, J, N, U>
{
    fn from(array: [[[TensorRank0; D]; D]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
