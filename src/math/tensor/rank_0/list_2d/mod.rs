use crate::math::{TensorList, TensorRank0List};

#[cfg(test)]
use crate::math::{Tensor, TensorRank0, test::ErrorTensor};

pub type TensorRank0List2D<const N: usize> = TensorList<TensorRank0List<N>, N>;

#[cfg(test)]
impl<const N: usize> ErrorTensor for TensorRank0List2D<N> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .filter(|&(&self_ij, &comparator_ij)| {
                        (self_ij / comparator_ij - 1.0).abs() >= epsilon
                            && (self_ij.abs() >= epsilon || comparator_ij.abs() >= epsilon)
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}
