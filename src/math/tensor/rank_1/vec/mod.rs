#[cfg(test)]
mod test;

use crate::math::{
    Jacobian, Solution, Tensor, TensorRank0, TensorRank1, TensorRank1List, TensorRank2SparseVec2D,
    TensorRank2SparseVec2DSymmetric, TensorRank2Vec2D, Vector, tensor::vec::TensorVector,
};
use std::{
    array::from_fn,
    mem::{forget, transmute},
    ops::{Div, Sub},
};

use crate::math::assert::FiniteDifference;

/// A vector of rank-1 tensors.
pub type TensorRank1Vec<const D: usize, const I: usize> = TensorVector<TensorRank1<D, I>>;

impl<const D: usize, const I: usize> TensorRank1Vec<D, I> {
    pub fn bounding_box(&self) -> TensorRank1List<D, I, 2> {
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
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| super::zero()).collect()
    }
}

impl<const D: usize, const I: usize, const N: usize> From<[[TensorRank0; D]; N]>
    for TensorRank1Vec<D, I>
{
    fn from(array: [[TensorRank0; D]; N]) -> Self {
        array.into_iter().map(TensorRank1::from).collect()
    }
}

impl<const D: usize, const I: usize> From<Vec<[TensorRank0; D]>> for TensorRank1Vec<D, I> {
    fn from(vec: Vec<[TensorRank0; D]>) -> Self {
        unsafe { transmute(vec) }
    }
}

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for Vec<[TensorRank0; D]> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        unsafe { transmute(tensor_rank_1_vec) }
    }
}

impl<const D: usize, const I: usize> From<Vec<Vec<TensorRank0>>> for TensorRank1Vec<D, I> {
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        vec.into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for Vec<Vec<TensorRank0>> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> TryFrom<[Vec<TensorRank0>; D]> for TensorRank1Vec<D, I> {
    type Error = String;
    fn try_from(vec_array: [Vec<TensorRank0>; D]) -> Result<Self, Self::Error> {
        let length = vec_array[0].len();
        if vec_array.iter().any(|vec| vec.len() != length) {
            Err("Vector length mismatch in type conversion".to_string())
        } else {
            Ok((0..length)
                .map(|j| TensorRank1::const_from(from_fn(|i| vec_array[i][j])))
                .collect())
        }
    }
}

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for [Vec<TensorRank0>; D] {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        let length = tensor_rank_1_vec.len();
        let mut output = from_fn(|_| Vec::with_capacity(length));
        tensor_rank_1_vec.into_iter().for_each(|tensor_rank_1| {
            output
                .iter_mut()
                .zip(tensor_rank_1)
                .for_each(|(entry, value)| entry.push(value))
        });
        output
    }
}

impl<const D: usize, const I: usize> From<&TensorRank1Vec<D, I>> for [Vec<TensorRank0>; D] {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, I>) -> Self {
        let length = tensor_rank_1_vec.len();
        let mut output = from_fn(|_| Vec::with_capacity(length));
        tensor_rank_1_vec.iter().for_each(|tensor_rank_1| {
            output
                .iter_mut()
                .zip(tensor_rank_1.iter())
                .for_each(|(entry, &value)| entry.push(value))
        });
        output
    }
}

impl<const D: usize> From<TensorRank1Vec<D, 0>> for TensorRank1Vec<D, 1> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, 0>) -> Self {
        let length = tensor_rank_1_vec.len();
        let pointer = tensor_rank_1_vec.as_ptr() as *mut TensorRank1<D, 1>;
        forget(tensor_rank_1_vec);
        unsafe { Self::from(Vec::from_raw_parts(pointer, length, length)) }
    }
}

impl<const D: usize> From<&TensorRank1Vec<D, 0>> for TensorRank1Vec<D, 1> {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, 0>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize> From<TensorRank1Vec<D, 1>> for TensorRank1Vec<D, 0> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, 1>) -> Self {
        let length = tensor_rank_1_vec.len();
        let pointer = tensor_rank_1_vec.as_ptr() as *mut TensorRank1<D, 0>;
        forget(tensor_rank_1_vec);
        unsafe { Self::from(Vec::from_raw_parts(pointer, length, length)) }
    }
}

impl<const D: usize> From<&TensorRank1Vec<D, 1>> for TensorRank1Vec<D, 0> {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, 1>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> From<Vector> for TensorRank1Vec<D, I> {
    fn from(vector: Vector) -> Self {
        let n = vector.len();
        if n.is_multiple_of(D) {
            let length = n / D;
            let pointer = vector.as_ptr() as *mut TensorRank1<D, I>;
            forget(vector);
            unsafe { Self::from(Vec::from_raw_parts(pointer, length, length)) }
        } else {
            panic!("Vector length mismatch.")
        }
    }
}

impl<const D: usize, const I: usize> Jacobian for TensorRank1Vec<D, I> {
    fn fill_into(self, vector: &mut Vector) {
        self.into_iter()
            .flatten()
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        self.into_iter()
            .flatten()
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn retain_from(self, retained: &[bool]) -> Vector {
        self.into_iter()
            .flatten()
            .zip(retained.iter())
            .filter(|(_, retained)| **retained)
            .map(|(entry, _)| entry)
            .collect()
    }
    fn zero_out(&mut self, indices: &[usize]) {
        indices
            .iter()
            .for_each(|index| self[index / D][index % D] = 0.0)
    }
}

impl<const D: usize, const I: usize> Solution for TensorRank1Vec<D, I> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(other.iter())
            .for_each(|(self_i, vector_i)| *self_i -= vector_i)
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .chain(other.iter_mut())
            .zip(vector)
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
    fn decrement_from_retained(&mut self, retained: &[bool], other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(retained.iter())
            .filter(|(_, retained_i)| **retained_i)
            .zip(other.iter())
            .for_each(|((self_i, _), vector_i)| *self_i -= vector_i)
    }
}

impl<const D: usize, const I: usize> Sub<Vector> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(a, self_a)| {
            self_a
                .iter_mut()
                .enumerate()
                .for_each(|(i, self_a_i)| *self_a_i -= vector[D * a + i])
        });
        self
    }
}

impl<const D: usize, const I: usize> Sub<&Vector> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(a, self_a)| {
            self_a
                .iter_mut()
                .enumerate()
                .for_each(|(i, self_a_i)| *self_a_i -= vector[D * a + i])
        });
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank2Vec2D<D, I, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank1Vec<D, J>;
    fn div(self, _tensor_rank_2_vec_2d: TensorRank2Vec2D<D, I, J>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank2SparseVec2D<D, I, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank1Vec<D, J>;
    fn div(self, _tensor_rank_2_sparse_vec_2d: TensorRank2SparseVec2D<D, I, J>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank2SparseVec2DSymmetric<D, I, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank1Vec<D, J>;
    fn div(
        self,
        _tensor_rank_2_sparse_symmetric_vec_2d: TensorRank2SparseVec2DSymmetric<D, I, J>,
    ) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const I: usize> FiniteDifference for TensorRank1Vec<D, I> {
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
