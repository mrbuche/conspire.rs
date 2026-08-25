#[cfg(test)]
mod test;
use crate::math::{ContractWith, Quantity, Square};
use crate::math::{Current, Factor, Flattened, Intermediate, Reference};
use crate::units::{Dimensionless, UnitDiv, UnitMul};

mod eigen;
mod inverse;
pub(crate) mod list;
pub(crate) mod list_2d;
mod logarithm;
mod power;
pub(crate) mod sparse_symmetric_vec_2d;
pub(crate) mod sparse_vec;
pub(crate) mod sparse_vec_2d;
pub(crate) mod vec;
pub(crate) mod vec_2d;

use std::{
    array::{IntoIter, from_fn},
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    Differentiate, Erase, Hessian, Jacobian, Rank2, Solution, SquareMatrix, Tensor, TensorArray,
    Vector,
    rank_0::TensorRank0,
    rank_1::{
        TensorRank1, list::TensorRank1List, relabel as relabel_rank_1, vec::TensorRank1Vec,
        zero as tensor_rank_1_zero,
    },
    rank_4::TensorRank4,
};
use crate::ABS_TOL;
use list_2d::TensorRank2List2D;
use vec_2d::TensorRank2Vec2D;

use crate::math::assert::FiniteDifference;

/// A *d*-dimensional tensor of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations.
#[repr(transparent)]
pub struct TensorRank2<const D: usize, I, J, U = Dimensionless>(
    [TensorRank1<D, J, U>; D],
    pub(super) PhantomData<I>,
);

impl<const D: usize, I, J, U> Clone for TensorRank2<D, I, J, U> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<const D: usize, I, J, U> Debug for TensorRank2<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const D: usize, I, J, U> PartialEq for TensorRank2<D, I, J, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J, U> Default for TensorRank2<D, I, J, U> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize, I, J, U> From<[[TensorRank0; D]; D]> for TensorRank2<D, I, J, U> {
    fn from(array: [[TensorRank0; D]; D]) -> Self {
        Self(from_fn(|i| TensorRank1::const_from(array[i])), PhantomData)
    }
}

impl<const D: usize, I, J, U> From<[[Quantity<U>; D]; D]> for TensorRank2<D, I, J, U> {
    fn from(array: [[Quantity<U>; D]; D]) -> Self {
        Self(from_fn(|i| TensorRank1(array[i], PhantomData)), PhantomData)
    }
}

impl<const D: usize, I, J, U> From<TensorRank2<D, I, J, U>> for [[TensorRank0; D]; D] {
    fn from(tensor_rank_2: TensorRank2<D, I, J, U>) -> Self {
        from_fn(|i| from_fn(|j| tensor_rank_2[i][j].value()))
    }
}

pub(crate) const fn get_levi_civita_parts<I, J, U>() -> [TensorRank2<3, I, J, U>; 3] {
    [
        TensorRank2(
            [
                tensor_rank_1_zero(),
                TensorRank1::const_from([0.0, 0.0, 1.0]),
                TensorRank1::const_from([0.0, -1.0, 0.0]),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                TensorRank1::const_from([0.0, 0.0, -1.0]),
                tensor_rank_1_zero(),
                TensorRank1::const_from([1.0, 0.0, 0.0]),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                TensorRank1::const_from([0.0, 1.0, 0.0]),
                TensorRank1::const_from([-1.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
    ]
}

pub(crate) const fn get_identity_1010_parts_1<I, J, U>() -> [TensorRank2<3, I, J, U>; 3] {
    [
        TensorRank2(
            [
                TensorRank1::const_from([1.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                TensorRank1::const_from([0.0, 1.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                TensorRank1::const_from([0.0, 0.0, 1.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
    ]
}

pub(crate) const fn get_identity_1010_parts_2<I, J, U>() -> [TensorRank2<3, I, J, U>; 3] {
    [
        TensorRank2(
            [
                tensor_rank_1_zero(),
                TensorRank1::const_from([1.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                tensor_rank_1_zero(),
                TensorRank1::const_from([0.0, 1.0, 0.0]),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                tensor_rank_1_zero(),
                TensorRank1::const_from([0.0, 0.0, 1.0]),
                tensor_rank_1_zero(),
            ],
            PhantomData,
        ),
    ]
}

pub(crate) const fn get_identity_1010_parts_3<I, J, U>() -> [TensorRank2<3, I, J, U>; 3] {
    [
        TensorRank2(
            [
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                TensorRank1::const_from([1.0, 0.0, 0.0]),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                TensorRank1::const_from([0.0, 1.0, 0.0]),
            ],
            PhantomData,
        ),
        TensorRank2(
            [
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                TensorRank1::const_from([0.0, 0.0, 1.0]),
            ],
            PhantomData,
        ),
    ]
}

/// The 3D identity, configurations (1, 1).
pub const IDENTITY: TensorRank2<3, Current, Current, Dimensionless> = TensorRank2(
    [
        TensorRank1::const_from([1.0, 0.0, 0.0]),
        TensorRank1::const_from([0.0, 1.0, 0.0]),
        TensorRank1::const_from([0.0, 0.0, 1.0]),
    ],
    PhantomData,
);

/// The 3D identity, configurations (0, 0).
pub const IDENTITY_00: TensorRank2<3, Reference, Reference, Dimensionless> = TensorRank2(
    [
        TensorRank1::const_from([1.0, 0.0, 0.0]),
        TensorRank1::const_from([0.0, 1.0, 0.0]),
        TensorRank1::const_from([0.0, 0.0, 1.0]),
    ],
    PhantomData,
);

/// The 3D identity, configurations (1, 0).
pub const IDENTITY_10: TensorRank2<3, Current, Reference, Dimensionless> = TensorRank2(
    [
        TensorRank1::const_from([1.0, 0.0, 0.0]),
        TensorRank1::const_from([0.0, 1.0, 0.0]),
        TensorRank1::const_from([0.0, 0.0, 1.0]),
    ],
    PhantomData,
);

/// The 3D identity, configurations (2, 2).
pub const IDENTITY_22: TensorRank2<3, Intermediate, Intermediate, Dimensionless> = TensorRank2(
    [
        TensorRank1::const_from([1.0, 0.0, 0.0]),
        TensorRank1::const_from([0.0, 1.0, 0.0]),
        TensorRank1::const_from([0.0, 0.0, 1.0]),
    ],
    PhantomData,
);

/// The 3D zero tensor, configurations (1, 1).
pub const ZERO: TensorRank2<3, Current, Current, Dimensionless> = TensorRank2(
    [
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
    ],
    PhantomData,
);

/// The 3D zero tensor, configurations (1, 0).
pub const ZERO_10: TensorRank2<3, Current, Reference, Dimensionless> = TensorRank2(
    [
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
    ],
    PhantomData,
);

impl<const D: usize, I, J, U> From<TensorRank1List<D, J, D, U>> for TensorRank2<D, I, J, U> {
    fn from(tensor_rank_1_list: TensorRank1List<D, J, D, U>) -> Self {
        tensor_rank_1_list.into_iter().collect()
    }
}

impl<const D: usize, I, J, U, V> From<(TensorRank1<D, I, U>, TensorRank1<D, J, V>)>
    for TensorRank2<D, I, J, <U as UnitMul<V>>::Output>
where
    U: UnitMul<V>,
{
    fn from((vector_a, vector_b): (TensorRank1<D, I, U>, TensorRank1<D, J, V>)) -> Self {
        vector_a
            .into_iter()
            .map(|vector_a_i| {
                vector_b
                    .iter()
                    .map(|vector_b_j| vector_a_i * vector_b_j)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, U, V> From<(TensorRank1<D, I, U>, &TensorRank1<D, J, V>)>
    for TensorRank2<D, I, J, <U as UnitMul<V>>::Output>
where
    U: UnitMul<V>,
{
    fn from((vector_a, vector_b): (TensorRank1<D, I, U>, &TensorRank1<D, J, V>)) -> Self {
        vector_a
            .into_iter()
            .map(|vector_a_i| {
                vector_b
                    .iter()
                    .map(|vector_b_j| vector_a_i * vector_b_j)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, U, V> From<(&TensorRank1<D, I, U>, TensorRank1<D, J, V>)>
    for TensorRank2<D, I, J, <U as UnitMul<V>>::Output>
where
    U: UnitMul<V>,
{
    fn from((vector_a, vector_b): (&TensorRank1<D, I, U>, TensorRank1<D, J, V>)) -> Self {
        vector_a
            .iter()
            .map(|vector_a_i| {
                vector_b
                    .iter()
                    .map(|vector_b_j| vector_a_i * vector_b_j)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, U, V> From<(&TensorRank1<D, I, U>, &TensorRank1<D, J, V>)>
    for TensorRank2<D, I, J, <U as UnitMul<V>>::Output>
where
    U: UnitMul<V>,
{
    fn from((vector_a, vector_b): (&TensorRank1<D, I, U>, &TensorRank1<D, J, V>)) -> Self {
        vector_a
            .iter()
            .map(|vector_a_i| {
                vector_b
                    .iter()
                    .map(|vector_b_j| vector_a_i * vector_b_j)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, U> From<Vec<Vec<TensorRank0>>> for TensorRank2<D, I, J, U> {
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        assert_eq!(vec.len(), D);
        vec.iter().for_each(|entry| assert_eq!(entry.len(), D));
        vec.into_iter()
            .map(|entry| entry.into_iter().collect())
            .collect()
    }
}

impl<const D: usize, I, J, U> From<TensorRank2<D, I, J, U>> for Vec<Vec<TensorRank0>> {
    fn from(tensor: TensorRank2<D, I, J, U>) -> Self {
        tensor
            .iter()
            .map(|entry| entry.iter().map(|entry_i| entry_i.value()).collect())
            .collect()
    }
}

impl<const D: usize, I, J, U> Display for TensorRank2<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, row)| write!(f, "{row},\n\x1B[u\x1B[{}B", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D)
    }
}

impl<const D: usize, I, J, U> FiniteDifference for TensorRank2<D, I, J, U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .filter(|&(&self_ij, &comparator_ij)| self_ij.differs(comparator_ij, epsilon))
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

impl<const D: usize, I, J, U> TensorRank2<D, I, J, U> {
    /// Asserts that the tensor carries the given unit.
    pub fn with_unit<V>(self) -> TensorRank2<D, I, J, V> {
        relabel(self.into_canonical())
    }
    pub(super) fn canonical(&self) -> &TensorRank2<D, Reference, Reference, Dimensionless> {
        unsafe {
            &*(self as *const Self as *const TensorRank2<D, Reference, Reference, Dimensionless>)
        }
    }
    fn canonical_mut(&mut self) -> &mut TensorRank2<D, Reference, Reference, Dimensionless> {
        unsafe {
            &mut *(self as *mut Self as *mut TensorRank2<D, Reference, Reference, Dimensionless>)
        }
    }
    fn into_canonical(self) -> TensorRank2<D, Reference, Reference, Dimensionless> {
        TensorRank2(self.0.map(recast), PhantomData)
    }
}

fn recast<const D: usize, I, J, U, V>(tensor_rank_1: TensorRank1<D, I, U>) -> TensorRank1<D, J, V> {
    TensorRank1(
        tensor_rank_1.0.map(|entry| Quantity::new(entry.value())),
        PhantomData,
    )
}

pub(super) fn relabel<const D: usize, I, J, U>(
    tensor_rank_2: TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank2<D, I, J, U> {
    TensorRank2(tensor_rank_2.0.map(recast), PhantomData)
}

impl<const D: usize> TensorRank2<D, Reference, Reference, Dimensionless> {
    fn as_array_core(&self) -> [[TensorRank0; D]; D] {
        let mut array = [[0.0; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry, tensor_rank_1)| {
                *entry = tensor_rank_1.as_array().map(|value| value.value())
            });
        array
    }
    fn identity_core() -> Self {
        (0..D)
            .map(|i| (0..D).map(|j| ((i == j) as u8) as TensorRank0).collect())
            .collect()
    }
    fn zero_core() -> Self {
        Self(from_fn(|_| TensorRank1::zero()), PhantomData)
    }
    fn add_assign_core(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2)
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
    fn add_assign_ref_core(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
    fn sub_assign_core(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2)
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
    fn sub_assign_ref_core(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
    fn mul_core(&self, tensor_rank_2: &Self) -> Self {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, I, J, U> TensorRank2<D, I, J, U> {
    /// Returns a raw pointer to the slice’s buffer.
    pub const fn as_ptr(&self) -> *const TensorRank1<D, J, U> {
        self.0.as_ptr()
    }
    /// Returns the rank-2 tensor reshaped as a rank-1 tensor.
    pub fn as_tensor_rank_1(&self) -> TensorRank1<9, Factor, U> {
        assert_eq!(D, 3);
        let mut tensor_rank_1 = TensorRank1::<9, Factor, U>::zero();
        self.iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .for_each(|(j, self_ij)| tensor_rank_1[3 * i + j] = *self_ij)
        });
        tensor_rank_1
    }
}

impl<const D: usize, I, J, U> Hessian for TensorRank2<D, I, J, U> {
    fn entry(&self, row: usize, column: usize) -> TensorRank0 {
        self[row][column].value()
    }
    fn quadratic_form(&self, vector: &Vector) -> TensorRank0 {
        self.iter()
            .zip(vector.iter())
            .map(|(self_i, vector_i)| {
                vector_i
                    * self_i
                        .iter()
                        .zip(vector.iter())
                        .map(|(self_ij, vector_j)| self_ij.value() * vector_j)
                        .sum::<TensorRank0>()
            })
            .sum()
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .into_iter()
                .enumerate()
                .for_each(|(j, self_ij)| square_matrix[i][j] = self_ij.value())
        })
    }
}

impl<const D: usize, I, J, U> Rank2 for TensorRank2<D, I, J, U> {
    type Transpose = TensorRank2<D, J, I, U>;
    fn deviatoric(&self) -> Self {
        Self::identity() * (self.trace().value() / -(D as TensorRank0)) + self
    }
    fn deviatoric_and_trace(&self) -> (Self, Quantity<U>) {
        let trace = self.trace();
        (
            Self::identity() * (trace.value() / -(D as TensorRank0)) + self,
            trace,
        )
    }
    fn is_diagonal(&self) -> bool {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .enumerate()
                    .map(|(j, self_ij)| (self_ij.value().abs() < ABS_TOL) as u8 * (i != j) as u8)
                    .sum::<u8>()
            })
            .sum::<u8>()
            == (D.pow(2) - D) as u8
    }
    fn is_identity(&self) -> bool {
        self.iter().enumerate().all(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .all(|(j, self_ij)| self_ij.value() == (i == j) as u8 as TensorRank0)
        })
    }
    fn is_symmetric(&self) -> bool {
        self.iter().enumerate().all(|(i, self_i)| {
            self_i
                .iter()
                .zip(self.iter())
                .all(|(self_ij, self_j)| self_ij == &self_j[i])
        })
    }
    fn squared_trace(&self) -> Quantity<Square<U>>
    where
        U: UnitMul<U>,
    {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .zip(self.iter())
                    .map(|(self_ij, self_j)| *self_ij * self_j[i])
                    .sum::<Quantity<Square<U>>>()
            })
            .sum()
    }
    fn trace(&self) -> Quantity<U> {
        self.iter().enumerate().map(|(i, self_i)| self_i[i]).sum()
    }
    fn transpose(&self) -> Self::Transpose {
        (0..D)
            .map(|i| (0..D).map(|j| self[j][i]).collect())
            // .map(|i| self.iter().map(|self_j| self_j[i]).collect())
            .collect()
    }
}

impl<const D: usize, I, J, U> Erase for TensorRank2<D, I, J, U> {
    type Erased = TensorRank2<D, Reference, Reference, Dimensionless>;
    fn erase(&self) -> &Self::Erased {
        self.canonical()
    }
}

impl<const D: usize, I, J, U> Tensor for TensorRank2<D, I, J, U> {
    type Item = TensorRank1<D, J, U>;
    type Unit = U;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn len(&self) -> usize {
        D
    }
    fn size(&self) -> usize {
        D * D
    }
}

impl<const D: usize, I, J, U> IntoIterator for TensorRank2<D, I, J, U> {
    type Item = TensorRank1<D, J, U>;
    type IntoIter = IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, I, J, U> TensorArray for TensorRank2<D, I, J, U> {
    type Array = [[TensorRank0; D]; D];
    type Item = TensorRank1<D, J, U>;
    fn as_array(&self) -> Self::Array {
        self.canonical().as_array_core()
    }
    fn identity() -> Self {
        relabel(TensorRank2::<D, Reference, Reference, Dimensionless>::identity_core())
    }
    fn zero() -> Self {
        relabel(TensorRank2::<D, Reference, Reference, Dimensionless>::zero_core())
    }
}

impl<const D: usize, I, J, U> Solution for TensorRank2<D, I, J, U> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(other.iter())
            .for_each(|(self_i, vector_i)| *self_i -= Quantity::new(*vector_i))
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector) {
        let mut values = vector.iter();
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(values.by_ref())
            .for_each(|(entry_i, vector_i)| *entry_i -= Quantity::new(*vector_i));
        other
            .iter_mut()
            .zip(values)
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
    fn decrement_from_retained(&mut self, retained: &[bool], other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(retained.iter())
            .filter(|(_, retained_i)| **retained_i)
            .zip(other.iter())
            .for_each(|((self_i, _), vector_i)| *self_i -= Quantity::new(*vector_i))
    }
}

impl<const D: usize, I, J, U> Jacobian for TensorRank2<D, I, J, U> {
    fn fill_into(&self, vector: &mut Vector) {
        self.iter()
            .flat_map(|entry| entry.iter())
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i.value())
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        self.into_iter()
            .flatten()
            .map(|entry| entry.value())
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn retain_from(self, retained: &[bool]) -> Vector {
        self.into_iter()
            .flatten()
            .zip(retained.iter())
            .filter(|(_, retained_i)| **retained_i)
            .map(|(entry, _)| entry.value())
            .collect()
    }
}

impl<const D: usize, I, J, U> Sub<Vector> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= Quantity::new(vector[D * i + j]))
        });
        self
    }
}

impl<const D: usize, I, J, U> Sub<&Vector> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= Quantity::new(vector[D * i + j]))
        });
        self
    }
}

impl<const D: usize, I, J, K, L, U> From<TensorRank4<D, I, J, K, L, U>>
    for TensorRank2<9, Factor, Flattened, U>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L, U>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> From<&TensorRank4<D, I, J, K, L, U>>
    for TensorRank2<9, Factor, Flattened, U>
{
    fn from(tensor_rank_4: &TensorRank4<D, I, J, K, L, U>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .clone()
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl<U> From<TensorRank2<3, Reference, Reference, U>>
    for TensorRank2<3, Intermediate, Intermediate, U>
{
    fn from(tensor_rank_2: TensorRank2<3, Reference, Reference, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<U> From<TensorRank2<3, Current, Current, U>>
    for TensorRank2<3, Intermediate, Intermediate, U>
{
    fn from(tensor_rank_2: TensorRank2<3, Current, Current, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<I, U> From<TensorRank2<3, I, Reference, U>> for TensorRank2<3, I, Intermediate, U> {
    fn from(tensor_rank_2: TensorRank2<3, I, Reference, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<I, U> From<TensorRank2<3, I, Current, U>> for TensorRank2<3, I, Reference, U> {
    fn from(tensor_rank_2: TensorRank2<3, I, Current, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<I, U> From<TensorRank2<3, I, Intermediate, U>> for TensorRank2<3, I, Reference, U> {
    fn from(tensor_rank_2: TensorRank2<3, I, Intermediate, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<J, U> From<TensorRank2<3, Reference, J, U>> for TensorRank2<3, Current, J, U> {
    fn from(tensor_rank_2: TensorRank2<3, Reference, J, U>) -> Self {
        Self(tensor_rank_2.0, PhantomData)
    }
}

impl<J, U> From<TensorRank2<3, Current, J, U>> for TensorRank2<3, Reference, J, U> {
    fn from(tensor_rank_2: TensorRank2<3, Current, J, U>) -> Self {
        Self(tensor_rank_2.0, PhantomData)
    }
}

impl<J, U> From<TensorRank2<3, Current, J, U>> for TensorRank2<3, Intermediate, J, U> {
    fn from(tensor_rank_2: TensorRank2<3, Current, J, U>) -> Self {
        Self(tensor_rank_2.0, PhantomData)
    }
}

impl<J, U> From<TensorRank2<3, Intermediate, J, U>> for TensorRank2<3, Current, J, U> {
    fn from(tensor_rank_2: TensorRank2<3, Intermediate, J, U>) -> Self {
        Self(tensor_rank_2.0, PhantomData)
    }
}

impl<J, U> From<&TensorRank2<3, Intermediate, J, U>> for &TensorRank2<3, Current, J, U> {
    fn from(tensor_rank_2: &TensorRank2<3, Intermediate, J, U>) -> Self {
        unsafe {
            transmute::<&TensorRank2<3, Intermediate, J, U>, &TensorRank2<3, Current, J, U>>(
                tensor_rank_2,
            )
        }
    }
}

impl<U> From<TensorRank2<3, Reference, Reference, U>> for TensorRank2<3, Current, Current, U> {
    fn from(tensor_rank_2: TensorRank2<3, Reference, Reference, U>) -> Self {
        Self(tensor_rank_2.0.map(recast), PhantomData)
    }
}

impl<const D: usize, I, J, U> From<Vector> for TensorRank2<D, I, J, U> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<const D: usize, I, J, U> FromIterator<TensorRank1<D, J, U>> for TensorRank2<D, I, J, U> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank1<D, J, U>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_2 = Self::zero();
        tensor_rank_2
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_2_i, value_i)| *tensor_rank_2_i = value_i);
        tensor_rank_2
    }
}

impl<const D: usize, I, J, U> Index<usize> for TensorRank2<D, I, J, U> {
    type Output = TensorRank1<D, J, U>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, I, J, U> IndexMut<usize> for TensorRank2<D, I, J, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, I, J, U> Sum for TensorRank2<D, I, J, U> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc += item;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl<'a, const D: usize, I, J, U> Sum<&'a Self> for TensorRank2<D, I, J, U> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::default(), |mut acc, item| {
            acc += item;
            acc
        })
    }
}

impl<const D: usize, I, J, U> Div<TensorRank0> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, U> Div<TensorRank0> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, U> Div<&TensorRank0> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, U> Div<&TensorRank0> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn div(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, U> DivAssign<TensorRank0> for TensorRank2<D, I, J, U> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, U> DivAssign<&TensorRank0> for TensorRank2<D, I, J, U> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, I, J, U> Mul<TensorRank0> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, U> Mul<TensorRank0> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, U> Mul<&TensorRank0> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, U> Mul<&TensorRank0> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, U> MulAssign<TensorRank0> for TensorRank2<D, I, J, U> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, U> MulAssign<&TensorRank0> for TensorRank2<D, I, J, U> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

fn canonical_rank_2_times_rank_1<const D: usize>(
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
    tensor_rank_1: &TensorRank1<D, Reference, Dimensionless>,
) -> TensorRank1<D, Reference, Dimensionless> {
    tensor_rank_2
        .iter()
        .map(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .zip(tensor_rank_1.iter())
                .map(|(tensor_rank_2_ij, tensor_rank_1_j)| tensor_rank_2_ij * tensor_rank_1_j)
                .sum::<Quantity>()
        })
        .collect()
}

impl<const D: usize, I, J, U, V> Mul<TensorRank1<D, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J, V>) -> Self::Output {
        relabel_rank_1(canonical_rank_2_times_rank_1(
            self.canonical(),
            tensor_rank_1.canonical(),
        ))
    }
}

impl<const D: usize, I, J, U, V> Mul<&TensorRank1<D, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J, V>) -> Self::Output {
        relabel_rank_1(canonical_rank_2_times_rank_1(
            self.canonical(),
            tensor_rank_1.canonical(),
        ))
    }
}

impl<const D: usize, I, J, U, V> Mul<TensorRank1<D, J, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J, V>) -> Self::Output {
        relabel_rank_1(canonical_rank_2_times_rank_1(
            self.canonical(),
            tensor_rank_1.canonical(),
        ))
    }
}

impl<const D: usize, I, J, U, V> Mul<&TensorRank1<D, J, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J, V>) -> Self::Output {
        relabel_rank_1(canonical_rank_2_times_rank_1(
            self.canonical(),
            tensor_rank_1.canonical(),
        ))
    }
}

impl<const D: usize, I, J, U> Add for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, I, J, U> Add<&Self> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: &Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, I, J, U> Add<TensorRank2<D, I, J, U>> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn add(self, mut tensor_rank_2: TensorRank2<D, I, J, U>) -> Self::Output {
        tensor_rank_2 += self;
        tensor_rank_2
    }
}

impl<const D: usize, I, J, U> AddAssign for TensorRank2<D, I, J, U> {
    fn add_assign(&mut self, tensor_rank_2: Self) {
        self.canonical_mut()
            .add_assign_core(tensor_rank_2.into_canonical());
    }
}

impl<const D: usize, I, J, U> AddAssign<&Self> for TensorRank2<D, I, J, U> {
    fn add_assign(&mut self, tensor_rank_2: &Self) {
        self.canonical_mut()
            .add_assign_ref_core(tensor_rank_2.canonical());
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2<D, J, K, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K, V>) -> Self::Output {
        relabel(self.canonical().mul_core(tensor_rank_2.canonical()))
    }
}

impl<const D: usize, I, J, K, U, V> Mul<&TensorRank2<D, J, K, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K, V>) -> Self::Output {
        relabel(self.canonical().mul_core(tensor_rank_2.canonical()))
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2<D, J, K, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K, V>) -> Self::Output {
        relabel(self.canonical().mul_core(tensor_rank_2.canonical()))
    }
}

impl<const D: usize, I, J, K, U, V> Mul<&TensorRank2<D, J, K, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K, V>) -> Self::Output {
        relabel(self.canonical().mul_core(tensor_rank_2.canonical()))
    }
}

impl<const D: usize, I, J, U, V> MulAssign<TensorRank2<D, J, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V, Output = U>,
{
    fn mul_assign(&mut self, tensor_rank_2: TensorRank2<D, J, J, V>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, I, J, U, V> MulAssign<&TensorRank2<D, J, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V, Output = U>,
{
    fn mul_assign(&mut self, tensor_rank_2: &TensorRank2<D, J, J, V>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, I, J, U> Sub for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, I, J, U> Sub<&Self> for TensorRank2<D, I, J, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: &Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, I, J, U> Sub<TensorRank2<D, I, J, U>> for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn sub(self, tensor_rank_2: TensorRank2<D, I, J, U>) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, I, J, U> Sub for &TensorRank2<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn sub(self, tensor_rank_2: Self) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, I, J, U> SubAssign for TensorRank2<D, I, J, U> {
    fn sub_assign(&mut self, tensor_rank_2: Self) {
        self.canonical_mut()
            .sub_assign_core(tensor_rank_2.into_canonical());
    }
}

impl<const D: usize, I, J, U> SubAssign<&Self> for TensorRank2<D, I, J, U> {
    fn sub_assign(&mut self, tensor_rank_2: &Self) {
        self.canonical_mut()
            .sub_assign_ref_core(tensor_rank_2.canonical());
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<TensorRank1List<D, J, W, V>>
    for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1List<D, I, W, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W, V>) -> Self::Output {
        tensor_rank_1_list
            .into_iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<&TensorRank1List<D, J, W, V>>
    for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1List<D, I, W, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W, V>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<TensorRank1List<D, J, W, V>>
    for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1List<D, I, W, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W, V>) -> Self::Output {
        tensor_rank_1_list
            .into_iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, const W: usize, U, V> Mul<&TensorRank1List<D, J, W, V>>
    for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1List<D, I, W, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W, V>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, U, V> Mul<TensorRank1Vec<D, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1Vec<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J, V>) -> Self::Output {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, U, V> Mul<&TensorRank1Vec<D, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1Vec<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J, V>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, U, V> Mul<TensorRank1Vec<D, J, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1Vec<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J, V>) -> Self::Output {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, U, V> Mul<&TensorRank1Vec<D, J, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1Vec<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J, V>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, I, J, K, const W: usize, const X: usize, U, V>
    Mul<TensorRank2List2D<D, J, K, W, X, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2List2D<D, I, K, W, X, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X, V>) -> Self::Output {
        tensor_rank_2_list_2d
            .into_iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .into_iter()
                    .map(|tensor_rank_2| &self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, const W: usize, const X: usize, U, V>
    Mul<TensorRank2List2D<D, J, K, W, X, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2List2D<D, I, K, W, X, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X, V>) -> Self::Output {
        tensor_rank_2_list_2d
            .into_iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .into_iter()
                    .map(|tensor_rank_2| self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2Vec2D<D, J, K, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2Vec2D<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K, V>) -> Self::Output {
        tensor_rank_2_list_2d
            .into_iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .into_iter()
                    .map(|tensor_rank_2| &self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2Vec2D<D, J, K, V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2Vec2D<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K, V>) -> Self::Output {
        tensor_rank_2_list_2d
            .into_iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .into_iter()
                    .map(|tensor_rank_2| self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

// Solving against a rank 4 divides the units, as it undoes multiplying by one.

#[allow(clippy::suspicious_arithmetic_impl)]
impl<I, J, K, L, U, V> Div<TensorRank4<3, I, J, K, L, V>> for &TensorRank2<3, I, J, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank2<3, K, L, <U as UnitDiv<V>>::Output>;
    fn div(self, tensor_rank_4: TensorRank4<3, I, J, K, L, V>) -> Self::Output {
        let tensor_rank_2: TensorRank2<9, Factor, Flattened, Dimensionless> =
            tensor_rank_4.with_unit::<Dimensionless>().into();
        let output_tensor_rank_1 = tensor_rank_2.inverse() * self.canonical().as_tensor_rank_1();
        let mut output = TensorRank2::<3, Reference, Reference, Dimensionless>::zero();
        output.iter_mut().enumerate().for_each(|(i, output_i)| {
            output_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, output_ij)| *output_ij = output_tensor_rank_1[3 * i + j])
        });
        relabel(output)
    }
}

impl<const D: usize, I, J, U, V> Mul<Quantity<V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() * quantity.value())
    }
}

impl<const D: usize, I, J, U, V> Mul<Quantity<V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.canonical() * quantity.value())
    }
}

impl<const D: usize, I, J, U, V> Mul<&Quantity<V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        self * *quantity
    }
}

impl<const D: usize, I, J, U, V> Mul<&Quantity<V>> for &TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        self * *quantity
    }
}

impl<const D: usize, I, J, U, V> Div<Quantity<V>> for TensorRank2<D, I, J, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() / quantity.value())
    }
}

impl<const D: usize, I, J, U, V> Div<Quantity<V>> for &TensorRank2<D, I, J, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.canonical() / quantity.value())
    }
}

impl<const D: usize, I, J, U, V> ContractWith<TensorRank2<D, I, J, V>> for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn contract_with(&self, tensor_rank_2: &TensorRank2<D, I, J, V>) -> Self::Output {
        Quantity::new(self.canonical().full_contraction(tensor_rank_2.canonical()))
    }
}

impl<const D: usize, I, J, U, T> Differentiate<T> for TensorRank2<D, I, J, U>
where
    U: UnitDiv<T>,
{
    type Derivative = TensorRank2<D, I, J, <U as UnitDiv<T>>::Output>;
}
