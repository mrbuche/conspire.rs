#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank1, TensorRank2, tensor::list::TensorList};
use std::{mem::transmute, ops::Mul};

#[cfg(test)]
use crate::math::{TensorRank0, tensor::test::ErrorTensor};

pub type TensorRank1List<const D: usize, const I: usize, const W: usize> =
    TensorList<TensorRank1<D, I>, W>;

impl From<TensorRank1List<3, 0, 3>> for TensorRank1List<3, 1, 3> {
    fn from(tensor_rank_1_list: TensorRank1List<3, 0, 3>) -> Self {
        unsafe {
            transmute::<TensorRank1List<3, 0, 3>, TensorRank1List<3, 1, 3>>(tensor_rank_1_list)
        }
    }
}

impl From<TensorRank1List<3, 0, 4>> for TensorRank1List<3, 1, 4> {
    fn from(tensor_rank_1_list: TensorRank1List<3, 0, 4>) -> Self {
        unsafe { transmute::<TensorRank1List<3, 0, 4>, Self>(tensor_rank_1_list) }
    }
}

impl From<TensorRank1List<3, 0, 8>> for TensorRank1List<3, 1, 8> {
    fn from(tensor_rank_1_list: TensorRank1List<3, 0, 8>) -> Self {
        unsafe { transmute::<TensorRank1List<3, 0, 8>, Self>(tensor_rank_1_list) }
    }
}

impl From<TensorRank1List<3, 0, 10>> for TensorRank1List<3, 1, 10> {
    fn from(tensor_rank_1_list: TensorRank1List<3, 0, 10>) -> Self {
        unsafe { transmute::<TensorRank1List<3, 0, 10>, Self>(tensor_rank_1_list) }
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| TensorRank2::dyad(self_entry, entry))
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| TensorRank2::dyad(self_entry, entry))
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| TensorRank2::dyad(self_entry, entry))
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
            .map(|(self_entry, entry)| TensorRank2::dyad(self_entry, entry))
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
                                && (entry_i.abs() >= epsilon
                                    || comparator_entry_i.abs() >= epsilon)
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
