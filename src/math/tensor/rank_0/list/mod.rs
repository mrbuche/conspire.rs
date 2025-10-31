pub mod vec;

#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank0, tensor::list::TensorList};
use std::ops::Mul;

pub type TensorRank0List<const W: usize> = TensorList<TensorRank0, W>;

impl<const W: usize> Mul for TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const W: usize> Mul<&Self> for TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: &Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const W: usize> Mul<TensorRank0List<W>> for &TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: TensorRank0List<W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const W: usize> Mul for &TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}
