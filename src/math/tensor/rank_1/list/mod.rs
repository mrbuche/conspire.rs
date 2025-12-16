#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank1, TensorRank2, tensor::list::TensorList};
use std::{mem::transmute, ops::Mul};

#[cfg(test)]
use crate::math::{TensorRank0, tensor::test::ErrorTensor};

pub type TensorRank1List<const D: usize, const I: usize, const W: usize> =
    TensorList<TensorRank1<D, I>, W>;

macro_rules! from_len {
    ($len:literal) => {
        impl From<TensorRank1List<3, 0, $len>> for TensorRank1List<3, 1, $len> {
            fn from(tensor_rank_1_list: TensorRank1List<3, 0, $len>) -> Self {
                unsafe { transmute::<TensorRank1List<3, 0, $len>, Self>(tensor_rank_1_list) }
            }
        }
    };
}
from_len!(3);
from_len!(4);
from_len!(5);
from_len!(6);
from_len!(8);
from_len!(10);

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1_list)
            .map(|(self_entry, entry)| (self_entry, entry).into())
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| (self_entry, entry).into())
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list)
            .map(|(self_entry, entry)| (self_entry, entry).into())
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| (self_entry, entry).into())
            .sum()
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const W: usize> ErrorTensor for TensorRank1List<D, I, W> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(entry, comparator_entry)| {
                entry
                    .iter()
                    .zip(comparator_entry.iter())
                    .filter(|&(&entry_i, &comparator_entry_i)| {
                        (entry_i / comparator_entry_i - 1.0).abs() >= epsilon
                            && (entry_i.abs() >= epsilon || comparator_entry_i.abs() >= epsilon)
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            let auxiliary = self
                .iter()
                .zip(comparator.iter())
                .map(|(entry, comparator_entry)| {
                    entry
                        .iter()
                        .zip(comparator_entry.iter())
                        .filter(|&(&entry_i, &comparator_entry_i)| {
                            (entry_i / comparator_entry_i - 1.0).abs() >= epsilon
                                && (entry_i - comparator_entry_i).abs() >= epsilon
                                && (entry_i.abs() >= epsilon || comparator_entry_i.abs() >= epsilon)
                        })
                        .count()
                })
                .sum::<usize>()
                > 0;
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}
