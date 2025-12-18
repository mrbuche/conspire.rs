#[cfg(test)]
mod test;

use crate::math::{TensorRank0, TensorRank4, tensor::list::TensorList};

pub type TensorRank4List<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const N: usize,
> = TensorList<TensorRank4<D, I, J, K, L>, N>;

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const N: usize>
    From<[[[[[TensorRank0; D]; D]; D]; D]; N]> for TensorRank4List<D, I, J, K, L, N>
{
    fn from(array: [[[[[TensorRank0; D]; D]; D]; D]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}
