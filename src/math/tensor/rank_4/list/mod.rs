#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank4, tensor::list::TensorList};
use crate::units::Dimensionless;

/// A list of rank-4 tensors.
pub type TensorRank4List<const D: usize, I, J, K, L, const N: usize, U = Dimensionless> =
    TensorList<TensorRank4<D, I, J, K, L, U>, N>;

impl<const D: usize, I, J, K, L, const N: usize, U> From<[[[[[TensorRank0; D]; D]; D]; D]; N]>
    for TensorRank4List<D, I, J, K, L, N, U>
{
    fn from(array: [[[[[TensorRank0; D]; D]; D]; D]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
