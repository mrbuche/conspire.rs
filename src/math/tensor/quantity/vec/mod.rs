use super::Quantity;
use crate::math::{
    Jacobian, Solution, Tensor, TensorRank0, Vector, assert::FiniteDifference,
    tensor::vec::TensorVector,
};
use crate::units::{Dimensionless, UnitDiv};
use std::ops::{Div, Sub};

use super::sparse_vec_2d::QuantitySparseVec2D;

/// A vector of quantities.
pub type QuantityVector<U = Dimensionless> = TensorVector<Quantity<U>>;

impl<U> QuantityVector<U> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| Quantity::new(0.0)).collect()
    }
}

impl<U> From<Vector> for QuantityVector<U> {
    fn from(vector: Vector) -> Self {
        vector.iter().map(|&entry| Quantity::new(entry)).collect()
    }
}

impl<U> Jacobian for QuantityVector<U> {
    fn fill_into(&self, vector: &mut Vector) {
        self.iter()
            .zip(vector.iter_mut())
            .for_each(|(self_a, vector_a)| *vector_a = self_a.value())
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        self.into_iter()
            .map(|entry| entry.value())
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(self_a, vector_a)| *vector_a = self_a)
    }
    fn retain_from(self, retained: &[bool]) -> Vector {
        self.into_iter()
            .zip(retained.iter())
            .filter(|(_, retained)| **retained)
            .map(|(entry, _)| entry.value())
            .collect()
    }
    fn zero_out(&mut self, indices: &[usize]) {
        indices
            .iter()
            .for_each(|&index| self[index] = Quantity::new(0.0))
    }
}

impl<U> Solution for QuantityVector<U> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(self_a, vector_a)| *self_a -= Quantity::new(*vector_a))
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_a, vector_a)| *self_a -= Quantity::new(*vector_a));
        other
            .iter_mut()
            .zip(vector.iter().skip(self.len()))
            .for_each(|(other_a, vector_a)| *other_a -= vector_a)
    }
    fn decrement_from_retained(&mut self, retained: &[bool], other: &Vector) {
        self.iter_mut()
            .zip(retained.iter())
            .filter(|(_, retained_a)| **retained_a)
            .zip(other.iter())
            .for_each(|((self_a, _), vector_a)| *self_a -= Quantity::new(*vector_a))
    }
}

impl<U> Sub<Vector> for QuantityVector<U> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_a, vector_a)| *self_a -= Quantity::new(*vector_a));
        self
    }
}

impl<U> Sub<&Vector> for QuantityVector<U> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_a, vector_a)| *self_a -= Quantity::new(*vector_a));
        self
    }
}

impl<U, V> Div<QuantitySparseVec2D<V>> for &QuantityVector<U>
where
    U: UnitDiv<V>,
{
    type Output = QuantityVector<<U as UnitDiv<V>>::Output>;
    fn div(self, _quantity_sparse_vec_2d: QuantitySparseVec2D<V>) -> Self::Output {
        unimplemented!(
            "A mesh-scale step wants the sparse solver the caller supplies, which a division has nowhere to hold."
        )
    }
}

impl<U> FiniteDifference for QuantityVector<U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .filter(|(entry, comparator_entry)| entry.differs(**comparator_entry, epsilon))
            .count();
        if error_count > 0 {
            let auxiliary = self
                .iter()
                .zip(comparator.iter())
                .filter(|(entry, comparator_entry)| {
                    entry.differs_severely(**comparator_entry, epsilon)
                })
                .count()
                > 0;
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}
