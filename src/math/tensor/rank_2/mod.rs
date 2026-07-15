#[cfg(test)]
mod test;

mod inverse;
pub mod list;
pub mod list_2d;
mod logarithm;
pub mod sparse_symmetric_vec_2d;
pub mod sparse_vec;
pub mod sparse_vec_2d;
pub mod vec;
pub mod vec_2d;

use std::{
    array::{IntoIter, from_fn},
    fmt::{self, Display, Formatter},
    iter::Sum,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    Hessian, Jacobian, Rank2, Solution, SquareMatrix, Tensor, TensorArray, Vector,
    rank_0::TensorRank0,
    rank_1::{TensorRank1, list::TensorRank1List, vec::TensorRank1Vec, zero as tensor_rank_1_zero},
    rank_4::TensorRank4,
};
use crate::ABS_TOL;
use list_2d::TensorRank2List2D;
use vec_2d::TensorRank2Vec2D;

#[cfg(test)]
use crate::math::assert::ErrorTensor;

/// A *d*-dimensional tensor of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRank2<const D: usize, const I: usize, const J: usize>([TensorRank1<D, J>; D]);

impl<const D: usize, const I: usize, const J: usize> Default for TensorRank2<D, I, J> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<[[TensorRank0; D]; D]>
    for TensorRank2<D, I, J>
{
    fn from(array: [[TensorRank0; D]; D]) -> Self {
        Self(from_fn(|i| array[i].into()))
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2<D, I, J>>
    for [[TensorRank0; D]; D]
{
    fn from(tensor_rank_2: TensorRank2<D, I, J>) -> Self {
        from_fn(|i| from_fn(|j| tensor_rank_2[i][j]))
    }
}

pub const fn get_levi_civita_parts<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3] {
    [
        TensorRank2([
            tensor_rank_1_zero(),
            TensorRank1::const_from([0.0, 0.0, 1.0]),
            TensorRank1::const_from([0.0, -1.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1::const_from([0.0, 0.0, -1.0]),
            tensor_rank_1_zero(),
            TensorRank1::const_from([1.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1::const_from([0.0, 1.0, 0.0]),
            TensorRank1::const_from([-1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_1<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            TensorRank1::const_from([1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            TensorRank1::const_from([0.0, 1.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            TensorRank1::const_from([0.0, 0.0, 1.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_2<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            tensor_rank_1_zero(),
            TensorRank1::const_from([1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            TensorRank1::const_from([0.0, 1.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            TensorRank1::const_from([0.0, 0.0, 1.0]),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_3<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1::const_from([1.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1::const_from([0.0, 1.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1::const_from([0.0, 0.0, 1.0]),
        ]),
    ]
}

pub const IDENTITY: TensorRank2<3, 1, 1> = TensorRank2([
    TensorRank1::const_from([1.0, 0.0, 0.0]),
    TensorRank1::const_from([0.0, 1.0, 0.0]),
    TensorRank1::const_from([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_00: TensorRank2<3, 0, 0> = TensorRank2([
    TensorRank1::const_from([1.0, 0.0, 0.0]),
    TensorRank1::const_from([0.0, 1.0, 0.0]),
    TensorRank1::const_from([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_10: TensorRank2<3, 1, 0> = TensorRank2([
    TensorRank1::const_from([1.0, 0.0, 0.0]),
    TensorRank1::const_from([0.0, 1.0, 0.0]),
    TensorRank1::const_from([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_22: TensorRank2<3, 2, 2> = TensorRank2([
    TensorRank1::const_from([1.0, 0.0, 0.0]),
    TensorRank1::const_from([0.0, 1.0, 0.0]),
    TensorRank1::const_from([0.0, 0.0, 1.0]),
]);

pub const ZERO: TensorRank2<3, 1, 1> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

pub const ZERO_10: TensorRank2<3, 1, 0> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

impl<const D: usize, const I: usize, const J: usize> From<TensorRank1List<D, J, D>>
    for TensorRank2<D, I, J>
{
    fn from(tensor_rank_1_list: TensorRank1List<D, J, D>) -> Self {
        tensor_rank_1_list.into_iter().collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<(TensorRank1<D, I>, TensorRank1<D, J>)>
    for TensorRank2<D, I, J>
{
    fn from((vector_a, vector_b): (TensorRank1<D, I>, TensorRank1<D, J>)) -> Self {
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

impl<const D: usize, const I: usize, const J: usize> From<(TensorRank1<D, I>, &TensorRank1<D, J>)>
    for TensorRank2<D, I, J>
{
    fn from((vector_a, vector_b): (TensorRank1<D, I>, &TensorRank1<D, J>)) -> Self {
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

impl<const D: usize, const I: usize, const J: usize> From<(&TensorRank1<D, I>, TensorRank1<D, J>)>
    for TensorRank2<D, I, J>
{
    fn from((vector_a, vector_b): (&TensorRank1<D, I>, TensorRank1<D, J>)) -> Self {
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

impl<const D: usize, const I: usize, const J: usize> From<(&TensorRank1<D, I>, &TensorRank1<D, J>)>
    for TensorRank2<D, I, J>
{
    fn from((vector_a, vector_b): (&TensorRank1<D, I>, &TensorRank1<D, J>)) -> Self {
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

impl<const D: usize, const I: usize, const J: usize> From<Vec<Vec<TensorRank0>>>
    for TensorRank2<D, I, J>
{
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        assert_eq!(vec.len(), D);
        vec.iter().for_each(|entry| assert_eq!(entry.len(), D));
        vec.into_iter()
            .map(|entry| entry.into_iter().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2<D, I, J>>
    for Vec<Vec<TensorRank0>>
{
    fn from(tensor: TensorRank2<D, I, J>) -> Self {
        tensor
            .iter()
            .map(|entry| entry.iter().copied().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Display for TensorRank2<D, I, J> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, row)| write!(f, "{row},\n\x1B[u\x1B[{}B", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D)
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor for TensorRank2<D, I, J> {
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

impl<const D: usize, const I: usize, const J: usize> TensorRank2<D, I, J> {
    /// Returns a raw pointer to the slice’s buffer.
    pub const fn as_ptr(&self) -> *const TensorRank1<D, J> {
        self.0.as_ptr()
    }
    /// Returns the rank-2 tensor reshaped as a rank-1 tensor.
    pub fn as_tensor_rank_1(&self) -> TensorRank1<9, 88> {
        assert_eq!(D, 3);
        let mut tensor_rank_1 = TensorRank1::<9, 88>::zero();
        self.iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .for_each(|(j, self_ij)| tensor_rank_1[3 * i + j] = *self_ij)
        });
        tensor_rank_1
    }
}

impl<const D: usize, const I: usize, const J: usize> Hessian for TensorRank2<D, I, J> {
    fn entry(&self, row: usize, column: usize) -> TensorRank0 {
        self[row][column]
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .into_iter()
                .enumerate()
                .for_each(|(j, self_ij)| square_matrix[i][j] = self_ij)
        })
    }
}

impl<const D: usize, const I: usize, const J: usize> Rank2 for TensorRank2<D, I, J> {
    type Transpose = TensorRank2<D, J, I>;
    fn deviatoric(&self) -> Self {
        Self::identity() * (self.trace() / -(D as TensorRank0)) + self
    }
    fn deviatoric_and_trace(&self) -> (Self, TensorRank0) {
        let trace = self.trace();
        (
            Self::identity() * (trace / -(D as TensorRank0)) + self,
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
                    .map(|(j, self_ij)| (self_ij.abs() < ABS_TOL) as u8 * (i != j) as u8)
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
                .all(|(j, self_ij)| self_ij == &((i == j) as u8 as TensorRank0))
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
    fn squared_trace(&self) -> TensorRank0 {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .zip(self.iter())
                    .map(|(self_ij, self_j)| self_ij * self_j[i])
                    .sum::<TensorRank0>()
            })
            .sum()
    }
    fn trace(&self) -> TensorRank0 {
        self.iter().enumerate().map(|(i, self_i)| self_i[i]).sum()
    }
    fn transpose(&self) -> Self::Transpose {
        (0..D)
            .map(|i| (0..D).map(|j| self[j][i]).collect())
            // .map(|i| self.iter().map(|self_j| self_j[i]).collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Tensor for TensorRank2<D, I, J> {
    type Item = TensorRank1<D, J>;
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

impl<const D: usize, const I: usize, const J: usize> IntoIterator for TensorRank2<D, I, J> {
    type Item = TensorRank1<D, J>;
    type IntoIter = IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, const I: usize, const J: usize> TensorArray for TensorRank2<D, I, J> {
    type Array = [[TensorRank0; D]; D];
    type Item = TensorRank1<D, J>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[0.0; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry, tensor_rank_1)| *entry = tensor_rank_1.as_array());
        array
    }
    fn identity() -> Self {
        (0..D)
            .map(|i| (0..D).map(|j| ((i == j) as u8) as TensorRank0).collect())
            .collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<const D: usize, const I: usize, const J: usize> Solution for TensorRank2<D, I, J> {
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
}

impl<const D: usize, const I: usize, const J: usize> Jacobian for TensorRank2<D, I, J> {
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
}

impl<const D: usize, const I: usize, const J: usize> Sub<Vector> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= vector[D * i + j])
        });
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Vector> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= vector[D * i + j])
        });
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<TensorRank4<D, I, J, K, L>> for TensorRank2<9, 88, 99>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<&TensorRank4<D, I, J, K, L>> for TensorRank2<9, 88, 99>
{
    fn from(tensor_rank_4: &TensorRank4<D, I, J, K, L>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .clone()
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl From<TensorRank2<3, 0, 0>> for TensorRank2<3, 2, 2> {
    fn from(tensor_rank_2: TensorRank2<3, 0, 0>) -> Self {
        unsafe { transmute::<TensorRank2<3, 0, 0>, TensorRank2<3, 2, 2>>(tensor_rank_2) }
    }
}

impl From<TensorRank2<3, 1, 1>> for TensorRank2<3, 2, 2> {
    fn from(tensor_rank_2: TensorRank2<3, 1, 1>) -> Self {
        unsafe { transmute::<TensorRank2<3, 1, 1>, TensorRank2<3, 2, 2>>(tensor_rank_2) }
    }
}

impl<const I: usize> From<TensorRank2<3, I, 0>> for TensorRank2<3, I, 2> {
    fn from(tensor_rank_2: TensorRank2<3, I, 0>) -> Self {
        unsafe { transmute::<TensorRank2<3, I, 0>, TensorRank2<3, I, 2>>(tensor_rank_2) }
    }
}

impl<const I: usize> From<TensorRank2<3, I, 1>> for TensorRank2<3, I, 0> {
    fn from(tensor_rank_2: TensorRank2<3, I, 1>) -> Self {
        unsafe { transmute::<TensorRank2<3, I, 1>, TensorRank2<3, I, 0>>(tensor_rank_2) }
    }
}

impl<const I: usize> From<TensorRank2<3, I, 2>> for TensorRank2<3, I, 0> {
    fn from(tensor_rank_2: TensorRank2<3, I, 2>) -> Self {
        unsafe { transmute::<TensorRank2<3, I, 2>, TensorRank2<3, I, 0>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 0, J>> for TensorRank2<3, 1, J> {
    fn from(tensor_rank_2: TensorRank2<3, 0, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 0, J>, TensorRank2<3, 1, J>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 1, J>> for TensorRank2<3, 0, J> {
    fn from(tensor_rank_2: TensorRank2<3, 1, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 1, J>, TensorRank2<3, 0, J>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 1, J>> for TensorRank2<3, 2, J> {
    fn from(tensor_rank_2: TensorRank2<3, 1, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 1, J>, TensorRank2<3, 2, J>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 2, J>> for TensorRank2<3, 1, J> {
    fn from(tensor_rank_2: TensorRank2<3, 2, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 2, J>, TensorRank2<3, 1, J>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<&TensorRank2<3, 2, J>> for &TensorRank2<3, 1, J> {
    fn from(tensor_rank_2: &TensorRank2<3, 2, J>) -> Self {
        unsafe { transmute::<&TensorRank2<3, 2, J>, &TensorRank2<3, 1, J>>(tensor_rank_2) }
    }
}

impl From<TensorRank2<3, 0, 0>> for TensorRank2<3, 1, 1> {
    fn from(tensor_rank_2: TensorRank2<3, 0, 0>) -> Self {
        unsafe { transmute::<TensorRank2<3, 0, 0>, TensorRank2<3, 1, 1>>(tensor_rank_2) }
    }
}

impl<const D: usize, const I: usize, const J: usize> From<Vector> for TensorRank2<D, I, J> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<const D: usize, const I: usize, const J: usize> FromIterator<TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank1<D, J>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_2 = Self::zero();
        tensor_rank_2
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_2_i, value_i)| *tensor_rank_2_i = value_i);
        tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> Index<usize> for TensorRank2<D, I, J> {
    type Output = TensorRank1<D, J>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize> IndexMut<usize> for TensorRank2<D, I, J> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize> Sum for TensorRank2<D, I, J> {
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

impl<'a, const D: usize, const I: usize, const J: usize> Sum<&'a Self> for TensorRank2<D, I, J> {
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

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<&TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<&TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn div(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<TensorRank0>
    for TensorRank2<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<&TensorRank0>
    for TensorRank2<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank0>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank0>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J>) -> Self::Output {
        self.into_iter()
            .map(|self_i| self_i * &tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J>) -> Self::Output {
        self.into_iter()
            .map(|self_i| self_i * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Add for TensorRank2<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<&Self> for TensorRank2<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: &Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<TensorRank2<D, I, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, J>;
    fn add(self, mut tensor_rank_2: TensorRank2<D, I, J>) -> Self::Output {
        tensor_rank_2 += self;
        tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign for TensorRank2<D, I, J> {
    fn add_assign(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2)
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign<&Self> for TensorRank2<D, I, J> {
    fn add_assign(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<&TensorRank2<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<&TensorRank2<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank2<D, J, J>>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_2: TensorRank2<D, J, J>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank2<D, J, J>>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_2: &TensorRank2<D, J, J>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Self> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: &Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<TensorRank2<D, I, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, J>;
    fn sub(self, tensor_rank_2: TensorRank2<D, I, J>) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn sub(self, tensor_rank_2: Self) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign for TensorRank2<D, I, J> {
    fn sub_assign(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2)
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign<&Self> for TensorRank2<D, I, J> {
    fn sub_assign(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .into_iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .into_iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .into_iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<TensorRank2List2D<D, J, K, W, X>> for TensorRank2<D, I, J>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<TensorRank2List2D<D, J, K, W, X>> for &TensorRank2<D, I, J>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2Vec2D<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2Vec2D<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K>) -> Self::Output {
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

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const I: usize, const J: usize, const K: usize, const L: usize> Div<TensorRank4<3, I, J, K, L>>
    for &TensorRank2<3, I, J>
{
    type Output = TensorRank2<3, K, L>;
    fn div(self, tensor_rank_4: TensorRank4<3, I, J, K, L>) -> Self::Output {
        let tensor_rank_2: TensorRank2<9, 88, 99> = tensor_rank_4.into();
        let output_tensor_rank_1 = tensor_rank_2.inverse() * self.as_tensor_rank_1();
        let mut output = TensorRank2::zero();
        output.iter_mut().enumerate().for_each(|(i, output_i)| {
            output_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, output_ij)| *output_ij = output_tensor_rank_1[3 * i + j])
        });
        output
    }
}
