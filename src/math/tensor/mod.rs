#[cfg(test)]
mod test;

pub(super) mod configuration;
pub(super) mod list;
pub(super) mod norm;
pub(super) mod quantity;
pub(super) mod rank_0;
pub(super) mod rank_1;
pub(super) mod rank_2;
pub(super) mod rank_3;
pub(super) mod rank_4;
pub(super) mod tuple;
pub(super) mod vec;

pub use configuration::{
    Auxiliary, Configuration, Current, Factor, Flattened, Intermediate, Projection, Reference,
};
pub use norm::Norm;
pub use quantity::{Is, Quantity};

use super::{SquareMatrix, Vector};
use crate::math::{Style, StyledError, styled_error};
use crate::units::{Dimensionless, Time, UnitMul};
use rank_0::{
    TensorRank0,
    list::{TensorRank0List, vec::TensorRank0ListVec},
};
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A scalar.
pub type Scalar = TensorRank0;

/// A vector of scalars.
pub type Scalars = Vector;

/// A list of scalars.
pub type ScalarList<const N: usize> = TensorRank0List<N>;

/// A vector of lists of scalars.
pub type ScalarListVec<const N: usize> = TensorRank0ListVec<N>;

/// Possible errors for tensors.
#[derive(PartialEq)]
pub enum TensorError {
    NotPositiveDefinite,
    SymmetricMatrixComplexEigenvalues,
}

impl StyledError for TensorError {
    fn message(&self, style: &Style) -> String {
        let h = style.headline;
        match self {
            Self::NotPositiveDefinite => format!("{h}Result is not positive definite."),
            Self::SymmetricMatrixComplexEigenvalues => {
                format!("{h}Symmetric matrix produced complex eigenvalues")
            }
        }
    }
}

styled_error!(TensorError);

/// A tensor that can be differentiated with respect to a variable of unit `T`.
///
/// The derivative is named by the tensor rather than passed in alongside it,
/// being the same tensor with its unit divided by the variable's. A tuple
/// computes each half's derivative on its own, so the pair of units its halves
/// carry never has to be taken apart.
///
/// The variable of integration need not be a time — an arclength or a load
/// parameter is just as ordinary — so it is named rather than assumed, with time
/// as the default for the common case.
pub trait Differentiate<T = Time>
where
    Self: Tensor,
{
    /// The derivative with respect to the variable of integration.
    type Derivative: Tensor;
}

/// The derivative of `Y` with respect to a variable of unit `T`.
///
/// Spelling the projection out at every use would crowd out the signatures it
/// appears in, since a tensor names a derivative for each variable it might be
/// differentiated against.
pub type Derivative<Y, T = Time> = <Y as Differentiate<T>>::Derivative;

/// The unit a quantity of unit `U` carries once squared.
///
/// A tensor squared against itself — its norm squared, the trace of its square,
/// the invariant built from the two — carries this rather than nothing. Spelling
/// the projection out at every use would crowd out the signatures, as it would
/// for a [`Derivative`].
pub type Square<U> = <U as UnitMul<U>>::Output;

/// The full contraction of two tensors whose units need not agree.
///
/// [`Tensor::full_contraction`] contracts a tensor with another of its own type
/// and gives a number. Contracting tensors of different units gives a quantity
/// whose unit is the product of theirs — a stress with a rate is a power
/// density — which is what erased views used to stand in for.
pub trait ContractWith<Rhs> {
    /// The quantity the contraction gives.
    type Output;
    /// Returns the full contraction with the other tensor.
    fn contract_with(&self, rhs: &Rhs) -> Self::Output;
}

/// Views a tensor with its configurations and unit discarded.
///
/// Contracting tensors of different units gives a quantity whose unit is the
/// product, which is not always one this library names. Generic code that only
/// wants the number contracts the erased views instead.
pub trait Erase {
    /// The tensor with its configurations and unit discarded.
    type Erased: Tensor;
    /// Views the tensor with its configurations and unit discarded.
    fn erase(&self) -> &Self::Erased;
}

impl Erase for TensorRank0 {
    type Erased = Self;
    fn erase(&self) -> &Self {
        self
    }
}

/// Common methods for solutions.
pub trait Solution
where
    Self: From<Vector> + Tensor,
{
    /// Decrements the solution from another vector.
    fn decrement_from(&mut self, other: &Vector);
    /// Decrements the solution chained with a vector from another vector.
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector);
    /// Decrements the solution from another vector on retained entries.
    fn decrement_from_retained(&mut self, _retained: &[bool], _other: &Vector) {
        unimplemented!()
    }
}

/// Common methods for Jacobians.
pub trait Jacobian
where
    Self:
        From<Vector> + Tensor + Sub<Vector, Output = Self> + for<'a> Sub<&'a Vector, Output = Self>,
{
    /// Fills the Jacobian into a vector.
    fn fill_into(&self, vector: &mut Vector);
    /// Fills the Jacobian chained with a vector into another vector.
    fn fill_into_chained(self, other: Vector, vector: &mut Vector);
    /// Return only the retained indices.
    fn retain_from(self, _retained: &[bool]) -> Vector {
        unimplemented!()
    }
    /// Zero out the specified indices.
    fn zero_out(&mut self, _indices: &[usize]) {
        unimplemented!()
    }
}

/// Common methods for Hessians.
pub trait Hessian
where
    Self: Tensor,
{
    /// The entry at the given (row, column) position.
    fn entry(&self, row: usize, column: usize) -> Scalar;
    /// Fills the Hessian into a square matrix.
    fn fill_into(self, square_matrix: &mut SquareMatrix);
    /// The quadratic form of the Hessian with a vector.
    ///
    /// ```math
    /// \mathbf{v}\cdot\mathbf{H}\cdot\mathbf{v}
    /// ```
    fn quadratic_form(&self, _vector: &Vector) -> Scalar {
        unimplemented!()
    }
    /// The Hessian applied to a vector.
    ///
    /// ```math
    /// \mathbf{H}\cdot\mathbf{v}
    /// ```
    ///
    /// This is all an iterative solve ever asks of a Hessian, so the default
    /// reaches every entry by position and leaves the sparse arrangements to
    /// say how to walk only the entries they keep.
    fn times(&self, vector: &Vector) -> Vector {
        (0..vector.len())
            .map(|row| {
                vector
                    .iter()
                    .enumerate()
                    .map(|(column, entry)| self.entry(row, column) * entry)
                    .sum()
            })
            .collect()
    }
    /// Return only the retained indices.
    fn retain_from(self, _retained: &[bool]) -> SquareMatrix {
        unimplemented!()
    }
}

/// Common methods for blocks of a Hessian.
pub trait HessianBlock {
    /// The entry of the block at the given row and column within it.
    fn entry(&self, row: usize, column: usize) -> TensorRank0;
    /// The number of rows the block occupies.
    fn height(&self) -> usize;
    /// The number of columns the block occupies.
    fn width(&self) -> usize;
    /// Fills the block into a matrix at the given row and column offsets.
    fn fill_into_block<M>(&self, matrix: &mut M, row: usize, column: usize)
    where
        M: IndexMut<usize, Output = Vector>;
}

/// Accumulates rank-2 blocks into a sparse Hessian-like structure.
///
/// Symmetric-safe: the caller guarantees `block` at (a, b) equals the
/// transpose of the (b, a) contribution, so implementors may store or
/// mirror as they see fit.
pub trait HessianAccumulate<const D: usize, I, U = Dimensionless> {
    fn accumulate(&mut self, a: usize, b: usize, block: rank_2::TensorRank2<D, I, I, U>);
}

/// Common methods for rank-2 tensors.
pub trait Rank2
where
    Self: Sized + Tensor,
{
    /// The type that is the transpose of the tensor.
    type Transpose;
    /// Returns the deviatoric component of the rank-2 tensor.
    fn deviatoric(&self) -> Self;
    /// Returns the deviatoric component and trace of the rank-2 tensor.
    fn deviatoric_and_trace(&self) -> (Self, Quantity<Self::Unit>);
    /// Checks whether the tensor is a diagonal tensor.
    fn is_diagonal(&self) -> bool;
    /// Checks whether the tensor is the identity tensor.
    fn is_identity(&self) -> bool;
    /// Checks whether the tensor is a symmetric tensor.
    fn is_symmetric(&self) -> bool;
    /// Returns the second invariant of the rank-2 tensor.
    fn second_invariant(&self) -> Quantity<Square<Self::Unit>>
    where
        Self::Unit: UnitMul<Self::Unit>,
    {
        let trace = self.trace();
        (trace * trace - self.squared_trace()) * 0.5
    }
    /// Returns the trace of the rank-2 tensor squared.
    fn squared_trace(&self) -> Quantity<Square<Self::Unit>>
    where
        Self::Unit: UnitMul<Self::Unit>;
    /// Returns the trace of the rank-2 tensor, which carries its unit.
    fn trace(&self) -> Quantity<Self::Unit>;
    /// Returns the transpose of the rank-2 tensor.
    fn transpose(&self) -> Self::Transpose;
}

/// Common methods for tensors.
#[allow(clippy::len_without_is_empty)]
pub trait Tensor
where
    for<'a> Self: Sized
        + Add<Self, Output = Self>
        + Add<&'a Self, Output = Self>
        + AddAssign
        + AddAssign<&'a Self>
        + Clone
        + Debug
        + Default
        + Display
        + Div<TensorRank0, Output = Self>
        // + Div<&'a TensorRank0, Output = Self>
        + DivAssign<TensorRank0>
        + DivAssign<&'a TensorRank0>
        + Mul<TensorRank0, Output = Self>
        // + Mul<&'a TensorRank0, Output = Self>
        + MulAssign<TensorRank0>
        + MulAssign<&'a TensorRank0>
        + Sub<Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + SubAssign
        + SubAssign<&'a Self>
        + Sum,
    Self::Item: Tensor,
{
    /// The type of item encountered when iterating over the tensor.
    type Item;
    /// The physical unit the tensor carries.
    type Unit;
    /// Returns number of nonzero entries given absolute and relative tolerances.
    fn error_count_zero(&self, tol_abs: Scalar, tol_rel: Scalar) -> Option<usize> {
        let error_count = self
            .iter()
            .filter_map(|entry| entry.error_count_zero(tol_abs, tol_rel))
            .sum();
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
    /// Returns number of different entries given absolute and relative tolerances.
    fn error_count(&self, other: &Self, tol_abs: Scalar, tol_rel: Scalar) -> Option<usize> {
        let error_count = self
            .iter()
            .zip(other.iter())
            .filter_map(|(self_entry, other_entry)| {
                self_entry.error_count(other_entry, tol_abs, tol_rel)
            })
            .sum();
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
    /// Returns the full contraction with another tensor.
    fn full_contraction(&self, tensor: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensor.iter())
            .map(|(self_entry, tensor_entry)| self_entry.full_contraction(tensor_entry))
            .sum()
    }
    /// Checks whether the tensor is the zero tensor.
    fn is_zero(&self) -> bool {
        self.iter().filter(|entry| !entry.is_zero()).count() == 0
    }
    /// Returns an iterator.
    ///
    /// The iterator yields all items from start to end. [Read more](https://doc.rust-lang.org/std/iter/)
    fn iter(&self) -> impl Iterator<Item = &Self::Item>;
    /// Returns an iterator that allows modifying each value.
    ///
    /// The iterator yields all items from start to end. [Read more](https://doc.rust-lang.org/std/iter/)
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item>;
    /// Returns the number of elements, also referred to as the ‘length’.
    fn len(&self) -> usize;
    /// Returns the tensor norm.
    fn norm(&self) -> Quantity<Self::Unit> {
        Quantity::new(self.full_contraction(self).sqrt())
    }
    /// Returns the infinity norm.
    fn norm_inf(&self) -> Quantity<Self::Unit> {
        Quantity::new(
            self.iter()
                .fold(0.0, |acc, entry| entry.norm_inf().value().max(acc)),
        )
    }
    /// Returns the L1 (Manhattan) norm.
    fn norm_l1(&self) -> Quantity<Self::Unit> {
        Quantity::new(
            self.iter()
                .fold(0.0, |acc, entry| acc + entry.norm_l1().value()),
        )
    }
    /// Returns the sum of p-th powers of absolute values (used internally by `norm_p`).
    fn norm_p_sum(&self, p: TensorRank0) -> TensorRank0 {
        self.iter()
            .fold(0.0, |acc, entry| acc + entry.norm_p_sum(p))
    }
    /// Returns the Minkowski (Lp) norm.
    fn norm_p(&self, p: TensorRank0) -> Quantity<Self::Unit> {
        Quantity::new(self.norm_p_sum(p).powf(1.0 / p))
    }
    /// Returns the tensor norm squared, which carries the square of its unit.
    fn norm_squared(&self) -> Quantity<Square<Self::Unit>>
    where
        Self::Unit: UnitMul<Self::Unit>,
    {
        Quantity::new(self.full_contraction(self))
    }
    /// Normalizes the tensor in place.
    fn normalize(&mut self) {
        *self /= self.norm().value()
    }
    /// Returns the total number of entries.
    fn size(&self) -> usize;
    /// Returns the positive difference of the two tensors.
    fn sub_abs(&self, other: &Self) -> Self {
        let mut difference = self.clone();
        difference
            .iter_mut()
            .zip(self.iter().zip(other.iter()))
            .for_each(|(entry, (self_entry, other_entry))| {
                *entry = self_entry.sub_abs(other_entry)
            });
        difference
    }
    /// Returns the relative difference of the two tensors.
    fn sub_rel(&self, other: &Self) -> Self {
        let mut difference = self.clone();
        difference
            .iter_mut()
            .zip(self.iter().zip(other.iter()))
            .for_each(|(entry, (self_entry, other_entry))| {
                *entry = self_entry.sub_rel(other_entry)
            });
        difference
    }
}

/// Common methods for tensors derived from arrays.
pub trait TensorArray {
    /// The type of array corresponding to the tensor.
    type Array;
    /// The type of item encountered when iterating over the tensor.
    type Item;
    /// Returns the tensor as an array.
    fn as_array(&self) -> Self::Array;
    /// Returns the identity tensor.
    fn identity() -> Self;
    /// Returns the zero tensor.
    fn zero() -> Self;
}

/// Common methods for tensors derived from Vec.
pub trait TensorVec
where
    Self: FromIterator<Self::Item> + Index<usize, Output = Self::Item> + IndexMut<usize>,
{
    /// The type of element encountered when iterating over the tensor.
    type Item;
    /// Moves all the elements of other into self, leaving other empty.
    fn append(&mut self, other: &mut Self);
    /// Returns the total number of elements the vector can hold without reallocating.
    fn capacity(&self) -> usize;
    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool;
    /// Constructs a new, empty Vec, not allocating until elements are pushed onto it.
    fn new() -> Self;
    /// Appends an element to the back of the Vec.
    fn push(&mut self, item: Self::Item);
    /// Removes an element from the Vec and returns it, shifting elements to the left.
    fn remove(&mut self, index: usize) -> Self::Item;
    /// Reserves capacity for at least additional more elements to be inserted in the given Vec.
    fn reserve(&mut self, additional: usize);
    /// Retains only the elements specified by the predicate.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&Self::Item) -> bool;
    /// Removes an element from the Vec and returns it, replacing it with the last element.
    fn swap_remove(&mut self, index: usize) -> Self::Item;
    /// Constructs a new, empty vector with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Self;
}
