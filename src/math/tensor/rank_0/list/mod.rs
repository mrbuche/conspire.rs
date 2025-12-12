pub mod vec;

#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank0, tensor::list::TensorList};
use std::ops::Mul;

#[cfg(test)]
use crate::math::tensor::test::ErrorTensor;

pub type TensorRank0List<const N: usize> = TensorList<TensorRank0, N>;

impl<const N: usize> Mul for TensorRank0List<N> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const N: usize> Mul<&Self> for TensorRank0List<N> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: &Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const N: usize> Mul<TensorRank0List<N>> for &TensorRank0List<N> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: TensorRank0List<N>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

impl<const N: usize> Mul for &TensorRank0List<N> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, entry)| self_entry * entry)
            .sum()
    }
}

#[cfg(test)]
impl<const N: usize> ErrorTensor for TensorRank0List<N> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .filter(|&(&self_i, &comparator_i)| {
                (self_i / comparator_i - 1.0).abs() >= epsilon
                    && (self_i.abs() >= epsilon || comparator_i.abs() >= epsilon)
            })
            .count();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}
