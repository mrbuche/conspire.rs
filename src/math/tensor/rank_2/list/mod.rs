pub mod vec;

#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank2, tensor::list::TensorList};

pub type TensorRank2List<const D: usize, const I: usize, const J: usize, const W: usize> =
    TensorList<TensorRank2<D, I, J>, W>;

impl<const D: usize, const I: usize, const J: usize, const N: usize>
    From<[[[TensorRank0; D]; D]; N]> for TensorRank2List<D, I, J, N>
{
    fn from(array: [[[TensorRank0; D]; D]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
