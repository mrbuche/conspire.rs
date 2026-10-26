#[cfg(test)]
mod test;

use crate::math::{Quantity, QuantityVector, Tensor, TensorVector, Vector};
use std::ops::{Div, Mul};

/// Residual modification to improve solver conditioning.
pub trait Precondition<F = Vector, X = Vector> {
    fn apply(&self, residual: &F) -> X;
}

/// Available preconditioning methods.
pub enum Preconditioning {
    /// A diagonal matrix.
    Diagonal(Vector),
    /// No preconditioning.
    None,
}

impl Precondition for Preconditioning {
    fn apply(&self, residual: &Vector) -> Vector {
        match self {
            Self::None => residual.clone(),
            Self::Diagonal(diagonal) => residual
                .iter()
                .zip(diagonal)
                .map(|(entry, scale)| entry / scale)
                .collect(),
        }
    }
}

impl<G, F, X> Precondition<F, X> for G
where
    G: Fn(&F) -> X,
{
    fn apply(&self, residual: &F) -> X {
        self(residual)
    }
}

/// A preconditioner that is a multiple of the identity.
#[derive(Clone, Copy, Debug)]
pub struct Scaled<V>(pub Quantity<V>);

impl<F, X, V> Precondition<F, X> for Scaled<V>
where
    F: Clone + Mul<Quantity<V>, Output = X>,
{
    fn apply(&self, residual: &F) -> X {
        residual.clone() * self.0
    }
}

/// A preconditioner that divides each item of a vector of tensors by an entry.
#[derive(Clone, Debug)]
pub struct Jacobi<V>(pub QuantityVector<V>);

impl<T, V, X> Precondition<TensorVector<T>, TensorVector<X>> for Jacobi<V>
where
    T: Tensor,
    X: Tensor,
    for<'a> &'a T: Div<Quantity<V>, Output = X>,
{
    fn apply(&self, residual: &TensorVector<T>) -> TensorVector<X> {
        residual
            .iter()
            .zip(&self.0)
            .map(|(item, &entry)| item / entry)
            .collect()
    }
}
