#[cfg(test)]
pub mod test;

pub mod rank_0;
pub mod rank_1;
pub mod rank_2;
pub mod rank_3;
pub mod rank_4;

use rank_0::TensorRank0;
use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
};

/// A value-to-value conversion that does not consume the input value.
///
/// This is as opposed to [`Into`](https://doc.rust-lang.org/std/convert/trait.Into.html), which consumes the input value.
pub trait Convert<T> {
    /// Converts this type into the (usually inferred) input type.
    fn convert(&self) -> T;
}

/// Possible errors for tensors.
#[derive(Debug)]
pub enum TensorError {
    NotPositiveDefinite,
}

impl PartialEq for TensorError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::NotPositiveDefinite => match other {
                Self::NotPositiveDefinite => true,
            },
        }
    }
}

/// Common methods for Hessians.
pub trait Hessian {
    /// Checks whether the Hessian is positive-definite.
    fn is_positive_definite(&self) -> bool;
}

/// Common methods for rank-2 tensors.
pub trait Rank2: Sized {
    /// The type that is the transpose of the tensor.
    type Transpose;
    /// Returns the Cholesky decomposition of the rank-2 tensor.
    fn cholesky_decomposition(&self) -> Result<Self, TensorError>;
    /// Returns the deviatoric component of the rank-2 tensor.
    fn deviatoric(&self) -> Self;
    /// Returns the deviatoric component and trace of the rank-2 tensor.
    fn deviatoric_and_trace(&self) -> (Self, TensorRank0);
    /// Checks whether the tensor is a diagonal tensor.
    fn is_diagonal(&self) -> bool;
    /// Checks whether the tensor is the identity tensor.
    fn is_identity(&self) -> bool;
    /// Returns the second invariant of the rank-2 tensor.
    fn second_invariant(&self) -> TensorRank0 {
        0.5 * (self.trace().powi(2) - self.squared_trace())
    }
    /// Returns the trace of the rank-2 tensor squared.
    fn squared_trace(&self) -> TensorRank0;
    /// Returns the trace of the rank-2 tensor.
    fn trace(&self) -> TensorRank0;
    /// Returns the transpose of the rank-2 tensor.
    fn transpose(&self) -> Self::Transpose;
}

/// Common methods for tensors.
pub trait Tensor
where
    for<'a> Self: Sized
        + Debug
        + Display
        + Add<Self, Output = Self>
        + Add<&'a Self, Output = Self>
        + AddAssign
        + AddAssign<&'a Self>
        + Div<TensorRank0, Output = Self>
        + Mul<TensorRank0, Output = Self>
        + Sub<Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + SubAssign
        + SubAssign<&'a Self>,
    Self::Item: Tensor,
{
    /// The type of item encountered when iterating over the tensor.
    type Item;
    /// Returns a copy.
    ///
    /// This method was implemented instead of the Copy trait to avoid unintended copy creations.
    fn copy(&self) -> Self;
    /// Returns the full contraction with another tensor.
    fn full_contraction(&self, tensor: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensor.iter())
            .map(|(self_entry, tensor_entry)| self_entry.full_contraction(tensor_entry))
            .sum()
    }
    /// Returns a reference to the entry at the specified indices.
    fn get_at(&self, _indices: &[usize]) -> &TensorRank0 {
        panic!("Need to implement get_at() for {:?}.", self)
    }
    /// Returns a mutable reference to the entry at the specified indices.
    fn get_at_mut(&mut self, _indices: &[usize]) -> &mut TensorRank0 {
        panic!("Need to implement get_at_mut() for {:?}.", self)
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
    /// Returns the tensor norm.
    fn norm(&self) -> TensorRank0 {
        self.norm_squared().sqrt()
    }
    /// Returns the tensor norm squared.
    fn norm_squared(&self) -> TensorRank0 {
        self.full_contraction(self)
    }
    /// Returns the tensor normalized.
    fn normalized(self) -> Self {
        let norm = self.norm();
        self / norm
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
    /// Returns a tensor given an array.
    fn new(array: Self::Array) -> Self;
    /// Returns the zero tensor.
    fn zero() -> Self;
}

/// Common methods for tensors derived from Vec.
pub trait TensorVec<'a> {
    /// The type of item encountered when iterating over the tensor.
    type Item;
    /// The type of slice corresponding to the tensor.
    type Slice;
    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool;
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    fn len(&self) -> usize;
    /// Returns a tensor given a slice.
    fn new(slice: Self::Slice) -> Self;
    /// Returns the zero tensor.
    fn zero(len: usize) -> Self;
}
