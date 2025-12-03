use crate::math::{TensorArray, TensorRank2, tensor::vec::TensorVector};

#[cfg(test)]
use crate::math::{Tensor, TensorRank0, tensor::test::ErrorTensor};

pub type TensorRank2Vec<const D: usize, const I: usize, const J: usize> =
    TensorVector<TensorRank2<D, I, J>>;

impl<const D: usize, const I: usize, const J: usize> TensorRank2Vec<D, I, J> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| TensorRank2::zero()).collect()
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor for TensorRank2Vec<D, I, J> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_a, comparator_a)| {
                self_a
                    .iter()
                    .zip(comparator_a.iter())
                    .map(|(self_a_i, comparator_a_i)| {
                        self_a_i
                            .iter()
                            .zip(comparator_a_i.iter())
                            .filter(|&(&self_a_ij, &comparator_a_ij)| {
                                (self_a_ij / comparator_a_ij - 1.0).abs() >= epsilon
                                    && (self_a_ij.abs() >= epsilon
                                        || comparator_a_ij.abs() >= epsilon)
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}
