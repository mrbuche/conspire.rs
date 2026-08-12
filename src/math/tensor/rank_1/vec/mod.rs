use crate::math::unit::{Dimensionless, UnitDiv};
use crate::math::{Current, Reference};
#[cfg(test)]
mod test;

use crate::math::{
    Jacobian, Solution, Tensor, TensorRank0, TensorRank1, TensorRank1List, TensorRank2SparseVec2D,
    TensorRank2SparseVec2DSymmetric, TensorRank2Vec2D, TensorVec, Vector,
    tensor::vec::TensorVector,
};
use std::{
    array::from_fn,
    mem::forget,
    ops::{Div, Sub},
};

use crate::math::assert::FiniteDifference;

/// A vector of rank-1 tensors.
pub type TensorRank1Vec<const D: usize, I, U = Dimensionless> = TensorVector<TensorRank1<D, I, U>>;

impl<const D: usize, I, U> TensorRank1Vec<D, I, U> {
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
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| super::zero()).collect()
    }
}

impl<const D: usize, I, const N: usize, U> From<[[TensorRank0; D]; N]> for TensorRank1Vec<D, I, U> {
    fn from(array: [[TensorRank0; D]; N]) -> Self {
        array.into_iter().map(TensorRank1::from).collect()
    }
}

impl<const D: usize, I, U> From<Vec<[TensorRank0; D]>> for TensorRank1Vec<D, I, U> {
    fn from(vec: Vec<[TensorRank0; D]>) -> Self {
        let (length, capacity) = (vec.len(), vec.capacity());
        let pointer = vec.as_ptr() as *mut TensorRank1<D, I, U>;
        forget(vec);
        unsafe { Self::from(Vec::from_raw_parts(pointer, length, capacity)) }
    }
}

impl<const D: usize, I, U> From<TensorRank1Vec<D, I, U>> for Vec<[TensorRank0; D]> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I, U>) -> Self {
        let vec = Vec::<TensorRank1<D, I, U>>::from(tensor_rank_1_vec);
        let (length, capacity) = (vec.len(), vec.capacity());
        let pointer = vec.as_ptr() as *mut [TensorRank0; D];
        forget(vec);
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

impl<const D: usize, I, U> From<Vec<Vec<TensorRank0>>> for TensorRank1Vec<D, I, U> {
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        vec.into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, I, U> From<TensorRank1Vec<D, I, U>> for Vec<Vec<TensorRank0>> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I, U>) -> Self {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, I, U> TryFrom<[Vec<TensorRank0>; D]> for TensorRank1Vec<D, I, U> {
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

impl<const D: usize, I, U> From<TensorRank1Vec<D, I, U>> for [Vec<TensorRank0>; D] {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I, U>) -> Self {
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

impl<const D: usize, I, U> From<&TensorRank1Vec<D, I, U>> for [Vec<TensorRank0>; D] {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, I, U>) -> Self {
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

impl<const D: usize, U> From<TensorRank1Vec<D, Reference, U>> for TensorRank1Vec<D, Current, U> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, Reference, U>) -> Self {
        let (length, capacity) = (tensor_rank_1_vec.len(), tensor_rank_1_vec.capacity());
        let pointer = tensor_rank_1_vec.as_ptr() as *mut TensorRank1<D, Current, U>;
        forget(tensor_rank_1_vec);
        unsafe { Self::from(Vec::from_raw_parts(pointer, length, capacity)) }
    }
}

impl<const D: usize, U> From<&TensorRank1Vec<D, Reference, U>> for TensorRank1Vec<D, Current, U> {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, Reference, U>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, U> From<TensorRank1Vec<D, Current, U>> for TensorRank1Vec<D, Reference, U> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, Current, U>) -> Self {
        let (length, capacity) = (tensor_rank_1_vec.len(), tensor_rank_1_vec.capacity());
        let pointer = tensor_rank_1_vec.as_ptr() as *mut TensorRank1<D, Reference, U>;
        forget(tensor_rank_1_vec);
        unsafe { Self::from(Vec::from_raw_parts(pointer, length, capacity)) }
    }
}

impl<const D: usize, U> From<&TensorRank1Vec<D, Current, U>> for TensorRank1Vec<D, Reference, U> {
    fn from(tensor_rank_1_vec: &TensorRank1Vec<D, Current, U>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, I, U> From<Vector> for TensorRank1Vec<D, I, U> {
    fn from(vector: Vector) -> Self {
        let n = vector.len();
        if !n.is_multiple_of(D) {
            panic!("Vector length mismatch.")
        } else if vector.capacity().is_multiple_of(D) {
            let (length, capacity) = (n / D, vector.capacity() / D);
            let pointer = vector.as_ptr() as *mut TensorRank1<D, I, U>;
            forget(vector);
            unsafe { Self::from(Vec::from_raw_parts(pointer, length, capacity)) }
        } else {
            (0..n / D)
                .map(|i| TensorRank1::const_from(from_fn(|j| vector[D * i + j])))
                .collect()
        }
    }
}

impl<const D: usize, I, U> Jacobian for TensorRank1Vec<D, I, U> {
    fn fill_into(&self, vector: &mut Vector) {
        self.iter()
            .flat_map(|entry| entry.iter())
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = *self_i)
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

impl<const D: usize, I, U> Solution for TensorRank1Vec<D, I, U> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(other.iter())
            .for_each(|(self_i, vector_i)| *self_i -= vector_i)
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .chain(other.iter_mut())
            .zip(vector.iter())
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

impl<const D: usize, I, U> Sub<Vector> for TensorRank1Vec<D, I, U> {
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

impl<const D: usize, I, U> Sub<&Vector> for TensorRank1Vec<D, I, U> {
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

impl<const D: usize, I, J, U> Div<TensorRank2Vec2D<D, I, J, U>> for &TensorRank1Vec<D, I, U> {
    type Output = TensorRank1Vec<D, J, U>;
    fn div(self, _tensor_rank_2_vec_2d: TensorRank2Vec2D<D, I, J, U>) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}

impl<const D: usize, I, J, U, V> Div<TensorRank2SparseVec2D<D, I, J, V>>
    for &TensorRank1Vec<D, I, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank1Vec<D, J, <U as UnitDiv<V>>::Output>;
    fn div(self, _tensor_rank_2_sparse_vec_2d: TensorRank2SparseVec2D<D, I, J, V>) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}

impl<const D: usize, I, J, U, V> Div<TensorRank2SparseVec2DSymmetric<D, I, J, V>>
    for &TensorRank1Vec<D, I, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank1Vec<D, J, <U as UnitDiv<V>>::Output>;
    fn div(
        self,
        _tensor_rank_2_sparse_symmetric_vec_2d: TensorRank2SparseVec2DSymmetric<D, I, J, V>,
    ) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}

impl<const D: usize, I, U> FiniteDifference for TensorRank1Vec<D, I, U> {
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
