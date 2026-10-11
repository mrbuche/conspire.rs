#[cfg(test)]
mod test;

use crate::math::{
    Jacobian, Matrix, Quantity, Solution, Tensor, TensorRank1, TensorTuple, TensorVector, Vector,
};
use std::{array::from_fn, ops::Mul, ops::Sub};

/// A vector of tensor tuples.
pub type TensorTupleVec<T1, T2> = TensorVector<TensorTuple<T1, T2>>;

impl<const D: usize, I, U1, U2> From<Vector>
    for TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>
{
    fn from(vector: Vector) -> Self {
        if !vector.len().is_multiple_of(D + 1) {
            panic!("Vector length mismatch.")
        }
        (0..vector.len() / (D + 1))
            .map(|a| {
                TensorTuple(
                    TensorRank1::const_from(from_fn(|i| vector[(D + 1) * a + i])),
                    Quantity::new(vector[(D + 1) * a + D]),
                )
            })
            .collect()
    }
}

impl<const D: usize, I, U1, U2> Jacobian for TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>> {
    fn fill_into(&self, vector: &mut Vector) {
        self.iter()
            .flat_map(|entry| {
                entry
                    .0
                    .iter()
                    .map(|entry_i| entry_i.value())
                    .chain([entry.1.value()])
            })
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        self.iter()
            .flat_map(|entry| {
                entry
                    .0
                    .iter()
                    .map(|entry_i| entry_i.value())
                    .chain([entry.1.value()])
            })
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn retain_from(self, retained: &[bool]) -> Vector {
        self.iter()
            .flat_map(|entry| {
                entry
                    .0
                    .iter()
                    .map(|entry_i| entry_i.value())
                    .chain([entry.1.value()])
            })
            .zip(retained.iter())
            .filter(|(_, retained)| **retained)
            .map(|(entry, _)| entry)
            .collect()
    }
    fn zero_out(&mut self, indices: &[usize]) {
        indices.iter().for_each(|index| {
            let (a, i) = (index / (D + 1), index % (D + 1));
            if i < D {
                self[a].0[i] = Quantity::new(0.0)
            } else {
                self[a].1 = Quantity::new(0.0)
            }
        })
    }
}

impl<const D: usize, I, U1, U2> Solution for TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut().enumerate().for_each(|(a, entry)| {
            entry
                .0
                .iter_mut()
                .enumerate()
                .for_each(|(i, entry_i)| *entry_i -= Quantity::new(other[(D + 1) * a + i]));
            entry.1 -= Quantity::new(other[(D + 1) * a + D])
        })
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector) {
        let size = (D + 1) * self.len();
        self.decrement_from(vector);
        other
            .iter_mut()
            .zip(vector.iter().skip(size))
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
    fn decrement_from_retained(&mut self, retained: &[bool], other: &Vector) {
        let mut values = other.iter();
        self.iter_mut().enumerate().for_each(|(a, entry)| {
            entry.0.iter_mut().enumerate().for_each(|(i, entry_i)| {
                if retained[(D + 1) * a + i] {
                    *entry_i -= Quantity::new(*values.next().unwrap())
                }
            });
            if retained[(D + 1) * a + D] {
                entry.1 -= Quantity::new(*values.next().unwrap())
            }
        })
    }
}

impl<const D: usize, I, U1, U2> Sub<Vector>
    for TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>
{
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.decrement_from(&vector);
        self
    }
}

impl<const D: usize, I, U1, U2> Sub<&Vector>
    for TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>
{
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.decrement_from(vector);
        self
    }
}

impl<const D: usize, I, U1, U2> Mul<&TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>>
    for &Vector
{
    type Output = f64;
    fn mul(self, tensor_tuple_vec: &TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>) -> f64 {
        tensor_tuple_vec
            .iter()
            .enumerate()
            .map(|(a, entry_a)| {
                entry_a
                    .0
                    .iter()
                    .map(|entry_a_i| entry_a_i.value())
                    .chain([entry_a.1.value()])
                    .enumerate()
                    .map(|(i, value)| self[(D + 1) * a + i] * value)
                    .sum::<f64>()
            })
            .sum()
    }
}

impl<const D: usize, I, U1, U2> Mul<&TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>>
    for &Matrix
{
    type Output = Vector;
    fn mul(self, tensor_tuple_vec: &TensorTupleVec<TensorRank1<D, I, U1>, Quantity<U2>>) -> Vector {
        self.iter()
            .map(|self_i| self_i * tensor_tuple_vec)
            .collect()
    }
}
