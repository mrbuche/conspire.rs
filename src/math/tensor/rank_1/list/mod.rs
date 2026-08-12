use crate::math::Dimensionless;
use crate::math::{Current, Projection, Reference};
#[cfg(test)]
mod test;

use crate::math::{
    CrossProduct, Tensor, TensorRank0, TensorRank1, TensorRank2, UnitMul, tensor::list::TensorList,
};
use std::ops::Mul;

use crate::math::assert::FiniteDifference;

/// A list of rank-1 tensors.
pub type TensorRank1List<const D: usize, I, const N: usize, U = Dimensionless> =
    TensorList<TensorRank1<D, I, U>, N>;

impl<const D: usize, I, const N: usize, U> TensorRank1List<D, I, N, U> {
    pub fn bounding_box(&self) -> TensorRank1List<D, I, 2, U> {
        self.iter()
            .skip(1)
            .fold(
                [self[0].clone(), self[0].clone()],
                |[mut min, mut max], entry| {
                    entry
                        .iter()
                        .zip(min.iter_mut().zip(max.iter_mut()))
                        .for_each(|(&entry_i, (min_i, max_i))| {
                            *min_i = min_i.min(entry_i);
                            *max_i = max_i.max(entry_i);
                        });
                    [min, max]
                },
            )
            .into()
    }
}

impl<I, U> TensorRank1List<3, I, 3, U>
where
    U: UnitMul<U>,
{
    /// Returns the scalar triple product, a number as a determinant is.
    pub fn scalar_triple_product(&self) -> TensorRank0 {
        &self[0] * self[1].cross(&self[2])
    }
}

impl<const D: usize, I, const N: usize, U> From<[[TensorRank0; D]; N]>
    for TensorRank1List<D, I, N, U>
{
    fn from(array: [[TensorRank0; D]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}

impl<const D: usize, const N: usize, U> From<TensorRank1List<D, Projection, N, U>>
    for TensorRank1List<D, Reference, N, U>
{
    fn from(tensor_rank_1_list: TensorRank1List<D, Projection, N, U>) -> Self {
        tensor_rank_1_list
            .into_iter()
            .map(|entry| entry.into())
            .collect()
    }
}

impl<const D: usize, const N: usize, U> From<TensorRank1List<D, Reference, N, U>>
    for TensorRank1List<D, Current, N, U>
{
    fn from(tensor_rank_1_list: TensorRank1List<D, Reference, N, U>) -> Self {
        tensor_rank_1_list
            .into_iter()
            .map(|entry| entry.into())
            .collect()
    }
}

impl<const D: usize, I, J, const W: usize, U> Mul<TensorRank1List<D, J, W, U>>
    for TensorRank1List<D, I, W, U>
{
    type Output = TensorRank2<D, I, J, U>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W, U>) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1_list)
            .map(|(self_entry, entry)| Self::Output::from((self_entry, entry)))
            .sum()
    }
}

impl<const D: usize, I, J, const W: usize, U> Mul<&TensorRank1List<D, J, W, U>>
    for TensorRank1List<D, I, W, U>
{
    type Output = TensorRank2<D, I, J, U>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W, U>) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| Self::Output::from((self_entry, entry)))
            .sum()
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<TensorRank1List<D, J, W, V>>
    for &TensorRank1List<D, I, W, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W, V>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list)
            .map(|(self_entry, entry)| Self::Output::from((self_entry, entry)))
            .sum()
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<&TensorRank1List<D, J, W, V>>
    for &TensorRank1List<D, I, W, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W, V>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, entry)| Self::Output::from((self_entry, entry)))
            .sum()
    }
}

impl<const D: usize, I, const W: usize, U> FiniteDifference for TensorRank1List<D, I, W, U> {
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
