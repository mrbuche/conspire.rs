use super::rank_2::relabel as relabel_rank_2;
use super::rank_3::relabel as relabel_rank_3;
use crate::math::Dimensionless;
use crate::math::UnitMul;
use crate::math::{Current, Intermediate, Reference};
use crate::math::{Quantity, UnitDiv};
#[cfg(test)]
mod test;

use crate::math::assert::FiniteDifference;

use std::{
    array::from_fn,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    Hessian, HessianBlock, Rank2, SquareMatrix, Tensor, TensorArray, Vector,
    rank_0::TensorRank0,
    rank_1::TensorRank1,
    rank_2::TensorRank2,
    rank_3::{TensorRank3, get_identity_1010_parts},
};

pub(crate) mod list;
pub(crate) mod vec;

impl<const D: usize, I, J, K, L, U> HessianBlock for TensorRank4<D, I, J, K, L, U> {
    fn entry(&self, row: usize, column: usize) -> TensorRank0 {
        self[row / D][row % D][column / D][column % D]
    }
    fn height(&self) -> usize {
        D * D
    }
    fn width(&self) -> usize {
        D * D
    }
    fn fill_into_block<M>(&self, matrix: &mut M, row: usize, column: usize)
    where
        M: IndexMut<usize, Output = Vector>,
    {
        self.iter().enumerate().for_each(|(i, self_i)| {
            self_i.iter().enumerate().for_each(|(j, self_ij)| {
                let matrix_row = &mut matrix[row + D * i + j];
                self_ij.iter().enumerate().for_each(|(k, self_ijk)| {
                    self_ijk
                        .iter()
                        .enumerate()
                        .for_each(|(l, self_ijkl)| matrix_row[column + D * k + l] = *self_ijkl)
                })
            })
        })
    }
}

/// A *d*-dimensional tensor of rank 4.
///
/// `D` is the dimension, `I`, `J`, `K`, `L` are the configurations.
#[repr(transparent)]
pub struct TensorRank4<const D: usize, I, J, K, L, U = Dimensionless>(
    [TensorRank3<D, J, K, L, U>; D],
    pub(super) PhantomData<I>,
);

impl<const D: usize, I, J, K, L, U> Clone for TensorRank4<D, I, J, K, L, U> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<const D: usize, I, J, K, L, U> Debug for TensorRank4<D, I, J, K, L, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const D: usize, I, J, K, L, U> PartialEq for TensorRank4<D, I, J, K, L, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J, K, L, U> TensorRank4<D, I, J, K, L, U> {
    /// Views the tensor with its configurations discarded, so that arithmetic is
    /// compiled once per dimension rather than once per configuration.
    /// Asserts that the tensor carries the given unit.
    pub fn with_unit<V>(self) -> TensorRank4<D, I, J, K, L, V> {
        relabel(self.into_canonical())
    }
    fn canonical(
        &self,
    ) -> &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
        unsafe {
            &*(self as *const Self
                as *const TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>)
        }
    }
    fn canonical_mut(
        &mut self,
    ) -> &mut TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
        unsafe {
            &mut *(self as *mut Self
                as *mut TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>)
        }
    }
    fn into_canonical(
        self,
    ) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
        unsafe {
            (&self as *const Self)
                .cast::<TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>>()
                .read()
        }
    }
}

fn relabel<const D: usize, I, J, K, L, U>(
    tensor: TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, I, J, K, L, U> {
    unsafe {
        (&tensor
            as *const TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>)
            .cast::<TensorRank4<D, I, J, K, L, U>>()
            .read()
    }
}

impl<const D: usize> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    fn as_array_core(&self) -> [[[[TensorRank0; D]; D]; D]; D] {
        let mut array = [[[[0.0; D]; D]; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_3, tensor_rank_3)| *entry_rank_3 = tensor_rank_3.as_array());
        array
    }
    fn zero_core() -> Self {
        Self(from_fn(|_| TensorRank3::zero()), PhantomData)
    }
    fn add_assign_core(&mut self, tensor: Self) {
        self.iter_mut()
            .zip(tensor)
            .for_each(|(self_i, tensor_i)| *self_i += tensor_i);
    }
    fn add_assign_ref_core(&mut self, tensor: &Self) {
        self.iter_mut()
            .zip(tensor.iter())
            .for_each(|(self_i, tensor_i)| *self_i += tensor_i);
    }
    fn sub_assign_core(&mut self, tensor: Self) {
        self.iter_mut()
            .zip(tensor)
            .for_each(|(self_i, tensor_i)| *self_i -= tensor_i);
    }
    fn sub_assign_ref_core(&mut self, tensor: &Self) {
        self.iter_mut()
            .zip(tensor.iter())
            .for_each(|(self_i, tensor_i)| *self_i -= tensor_i);
    }
}

impl<const D: usize, I, J, K, L, U> Default for TensorRank4<D, I, J, K, L, U> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize, I, J, K, L, U> From<[[[[TensorRank0; D]; D]; D]; D]>
    for TensorRank4<D, I, J, K, L, U>
{
    fn from(array: [[[[TensorRank0; D]; D]; D]; D]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}

/// The 3D rank-4 identity, configurations (1, 0, 1, 0).
pub const IDENTITY_1010: TensorRank4<3, Current, Reference, Current, Reference, Dimensionless> =
    TensorRank4(get_identity_1010_parts(), PhantomData);

impl<I, J, K, U> From<TensorRank4<3, I, J, K, Reference, U>>
    for TensorRank4<3, I, J, K, Intermediate, U>
{
    fn from(tensor_rank_4: TensorRank4<3, I, J, K, Reference, U>) -> Self {
        unsafe {
            transmute::<
                TensorRank4<3, I, J, K, Reference, U>,
                TensorRank4<3, I, J, K, Intermediate, U>,
            >(tensor_rank_4)
        }
    }
}

impl<J, L, U> From<TensorRank4<3, Current, J, Current, L, U>>
    for TensorRank4<3, Intermediate, J, Intermediate, L, U>
{
    fn from(tensor_rank_4: TensorRank4<3, Current, J, Current, L, U>) -> Self {
        unsafe {
            transmute::<
                TensorRank4<3, Current, J, Current, L, U>,
                TensorRank4<3, Intermediate, J, Intermediate, L, U>,
            >(tensor_rank_4)
        }
    }
}

impl<I, K, U> From<TensorRank4<3, I, Reference, K, Reference, U>>
    for TensorRank4<3, I, Intermediate, K, Intermediate, U>
{
    fn from(tensor_rank_4: TensorRank4<3, I, Reference, K, Reference, U>) -> Self {
        unsafe {
            transmute::<
                TensorRank4<3, I, Reference, K, Reference, U>,
                TensorRank4<3, I, Intermediate, K, Intermediate, U>,
            >(tensor_rank_4)
        }
    }
}

impl<K, U> From<TensorRank4<3, Reference, Reference, K, Reference, U>>
    for TensorRank4<3, Intermediate, Intermediate, K, Intermediate, U>
{
    fn from(tensor_rank_4: TensorRank4<3, Reference, Reference, K, Reference, U>) -> Self {
        unsafe {
            transmute::<
                TensorRank4<3, Reference, Reference, K, Reference, U>,
                TensorRank4<3, Intermediate, Intermediate, K, Intermediate, U>,
            >(tensor_rank_4)
        }
    }
}

impl<const D: usize, I, J, K, L, U> From<Vec<Vec<Vec<Vec<TensorRank0>>>>>
    for TensorRank4<D, I, J, K, L, U>
{
    fn from(vec_rank_4: Vec<Vec<Vec<Vec<TensorRank0>>>>) -> Self {
        vec_rank_4
            .into_iter()
            .map(|vec_rank_3| {
                vec_rank_3
                    .into_iter()
                    .map(|vec_rank_2| {
                        vec_rank_2
                            .into_iter()
                            .map(|vec_rank_1| vec_rank_1.into_iter().collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> From<TensorRank4<D, I, J, K, L, U>>
    for Vec<Vec<Vec<Vec<TensorRank0>>>>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L, U>) -> Self {
        tensor_rank_4
            .iter()
            .map(|tensor_rank_3| {
                tensor_rank_3
                    .iter()
                    .map(|tensor_rank_2| {
                        tensor_rank_2
                            .iter()
                            .map(|tensor_rank_1| tensor_rank_1.iter().copied().collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> From<TensorRank4<D, I, J, K, L, U>> for Vec<TensorRank0> {
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L, U>) -> Self {
        tensor_rank_4
            .iter()
            .flat_map(|tensor_rank_3| {
                tensor_rank_3.iter().flat_map(|tensor_rank_2| {
                    tensor_rank_2
                        .iter()
                        .flat_map(|tensor_rank_1| tensor_rank_1.iter().copied())
                })
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> From<TensorRank4<D, I, J, K, L, U>> for Vector {
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L, U>) -> Self {
        tensor_rank_4
            .iter()
            .flat_map(|tensor_rank_3| {
                tensor_rank_3.iter().flat_map(|tensor_rank_2| {
                    tensor_rank_2
                        .iter()
                        .flat_map(|tensor_rank_1| tensor_rank_1.iter().copied())
                })
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> Display for TensorRank4<D, I, J, K, L, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, entry)| write!(f, "{entry},\n\x1B[u\x1B[{}B\x1B[2D", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D + 2)
    }
}

impl<const D: usize, I, J, K, L, U> FiniteDifference for TensorRank4<D, I, J, K, L, U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .map(|(self_ij, comparator_ij)| {
                        self_ij
                            .iter()
                            .zip(comparator_ij.iter())
                            .map(|(self_ijk, comparator_ijk)| {
                                self_ijk
                                    .iter()
                                    .zip(comparator_ijk.iter())
                                    .filter(|&(&self_ijkl, &comparator_ijkl)| {
                                        (self_ijkl / comparator_ijkl - 1.0).abs() >= epsilon
                                            && (self_ijkl.abs() >= epsilon
                                                || comparator_ijkl.abs() >= epsilon)
                                    })
                                    .count()
                            })
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            let auxiliary = self
                .iter()
                .zip(comparator.iter())
                .map(|(self_i, comparator_i)| {
                    self_i
                        .iter()
                        .zip(comparator_i.iter())
                        .map(|(self_ij, comparator_ij)| {
                            self_ij
                                .iter()
                                .zip(comparator_ij.iter())
                                .map(|(self_ijk, comparator_ijk)| {
                                    self_ijk
                                        .iter()
                                        .zip(comparator_ijk.iter())
                                        .filter(|&(&self_ijkl, &comparator_ijkl)| {
                                            (self_ijkl / comparator_ijkl - 1.0).abs() >= epsilon
                                                && (self_ijkl - comparator_ijkl).abs() >= epsilon
                                                && (self_ijkl.abs() >= epsilon
                                                    || comparator_ijkl.abs() >= epsilon)
                                        })
                                        .count()
                                })
                                .sum::<usize>()
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
                > 0;
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, I, J, K, L, U> TensorRank4<D, I, J, K, L, U> {
    pub fn dyad_ij_kl<U1, U2>(
        tensor_rank_2_a: &TensorRank2<D, I, J, U1>,
        tensor_rank_2_b: &TensorRank2<D, K, L, U2>,
    ) -> Self
    where
        U1: UnitMul<U2, Output = U>,
    {
        relabel(canonical_dyad_ij_kl(
            tensor_rank_2_a.canonical(),
            tensor_rank_2_b.canonical(),
        ))
    }
    pub fn dyad_ik_jl<U1, U2>(
        tensor_rank_2_a: &TensorRank2<D, I, K, U1>,
        tensor_rank_2_b: &TensorRank2<D, J, L, U2>,
    ) -> Self
    where
        U1: UnitMul<U2, Output = U>,
    {
        relabel(canonical_dyad_ik_jl(
            tensor_rank_2_a.canonical(),
            tensor_rank_2_b.canonical(),
        ))
    }
    pub fn dyad_il_jk<U1, U2>(
        tensor_rank_2_a: &TensorRank2<D, I, L, U1>,
        tensor_rank_2_b: &TensorRank2<D, J, K, U2>,
    ) -> Self
    where
        U1: UnitMul<U2, Output = U>,
    {
        relabel(canonical_dyad_il_jk(
            tensor_rank_2_a.canonical(),
            tensor_rank_2_b.canonical(),
        ))
    }
    pub fn dyad_il_kj<U1, U2>(
        tensor_rank_2_a: &TensorRank2<D, I, L, U1>,
        tensor_rank_2_b: &TensorRank2<D, K, J, U2>,
    ) -> Self
    where
        U1: UnitMul<U2, Output = U>,
    {
        Self::dyad_il_jk(tensor_rank_2_a, &(tensor_rank_2_b.transpose()))
    }
}

impl<const D: usize, I, J, K, L, U> Hessian for TensorRank4<D, I, J, K, L, U> {
    fn entry(&self, row: usize, column: usize) -> TensorRank0 {
        self[row / D][row % D][column / D][column % D]
    }
    fn quadratic_form(&self, vector: &Vector) -> TensorRank0 {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .enumerate()
                    .map(|(j, self_ij)| {
                        vector[D * i + j]
                            * self_ij
                                .iter()
                                .enumerate()
                                .map(|(k, self_ijk)| {
                                    self_ijk
                                        .iter()
                                        .enumerate()
                                        .map(|(l, self_ijkl)| self_ijkl * vector[D * k + l])
                                        .sum::<TensorRank0>()
                                })
                                .sum::<TensorRank0>()
                    })
                    .sum::<TensorRank0>()
            })
            .sum()
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(i, self_i)| {
            self_i.into_iter().enumerate().for_each(|(j, self_ij)| {
                self_ij.into_iter().enumerate().for_each(|(k, self_ijk)| {
                    self_ijk
                        .into_iter()
                        .enumerate()
                        .for_each(|(l, self_ijkl)| square_matrix[D * i + j][D * k + l] = self_ijkl)
                })
            })
        })
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        (0..D * D)
            .filter(|&row| retained[row])
            .map(|row| {
                (0..D * D)
                    .filter(|&column| retained[column])
                    .map(|column| self[row / D][row % D][column / D][column % D])
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> Tensor for TensorRank4<D, I, J, K, L, U> {
    type Item = TensorRank3<D, J, K, L, U>;
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
        D * D * D * D
    }
}

impl<const D: usize, I, J, K, L, U> IntoIterator for TensorRank4<D, I, J, K, L, U> {
    type Item = TensorRank3<D, J, K, L, U>;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, I, J, K, L, U> TensorArray for TensorRank4<D, I, J, K, L, U> {
    type Array = [[[[TensorRank0; D]; D]; D]; D];
    type Item = TensorRank3<D, J, K, L, U>;
    fn as_array(&self) -> Self::Array {
        self.canonical().as_array_core()
    }
    fn identity() -> Self {
        relabel(canonical_dyad_ij_kl(
            &TensorRank2::identity(),
            &TensorRank2::identity(),
        ))
    }
    fn zero() -> Self {
        relabel(TensorRank4::<
            D,
            Reference,
            Reference,
            Reference,
            Reference,
            Dimensionless,
        >::zero_core())
    }
}

impl<const D: usize, I, J, K, L, U> FromIterator<TensorRank3<D, J, K, L, U>>
    for TensorRank4<D, I, J, K, L, U>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank3<D, J, K, L, U>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_4 = Self::zero();
        tensor_rank_4
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_4_i, value_i)| *tensor_rank_4_i = value_i);
        tensor_rank_4
    }
}

impl<const D: usize, I, J, K, L, U> Index<usize> for TensorRank4<D, I, J, K, L, U> {
    type Output = TensorRank3<D, J, K, L, U>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, I, J, K, L, U> IndexMut<usize> for TensorRank4<D, I, J, K, L, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, I, J, K, L, U> Sum for TensorRank4<D, I, J, K, L, U> {
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

impl<'a, const D: usize, I, J, K, L, U> Sum<&'a Self> for TensorRank4<D, I, J, K, L, U> {
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

/// Transforms all four indices of a rank-4 tensor.
///
/// Contracts each index of `self` with the first index of a corresponding rank-2 tensor.
pub trait ContractAllWithFirst<TIM, TJN, TKO, TLP> {
    type Output;
    fn contract_all_with_first(
        self,
        object_a: TIM,
        object_b: TJN,
        object_c: TKO,
        object_d: TLP,
    ) -> Self::Output;
}

impl<const D: usize, I, J, K, L, M, N, O, P, U>
    ContractAllWithFirst<
        &TensorRank2<D, I, M, U>,
        &TensorRank2<D, J, N, U>,
        &TensorRank2<D, K, O, U>,
        &TensorRank2<D, L, P, U>,
    > for TensorRank4<D, I, J, K, L, U>
{
    type Output = TensorRank4<D, M, N, O, P, U>;
    fn contract_all_with_first(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M, U>,
        tensor_rank_2_b: &TensorRank2<D, J, N, U>,
        tensor_rank_2_c: &TensorRank2<D, K, O, U>,
        tensor_rank_2_d: &TensorRank2<D, L, P, U>,
    ) -> Self::Output {
        // One index at a time. Transforming all four at once sums over every
        // index of the input for every index of the output, which is the
        // dimension cubed more arithmetic than doing them in sequence.
        let first = canonical_transform_first(self.canonical(), tensor_rank_2_a.canonical());
        let second = canonical_contract_second_with_first(&first, tensor_rank_2_b.canonical());
        let third = canonical_contract_third_with_first(&second, tensor_rank_2_c.canonical());
        relabel(canonical_transform_fourth(
            &third,
            tensor_rank_2_d.canonical(),
        ))
    }
}

/// Transforms the first, third, and fourth indices of a rank-4 tensor.
///
/// Contracts each with the first index of a corresponding rank-2 tensor;
/// the second index of `self` is left untouched.
pub trait ContractFirstThirdFourthWithFirst<TIM, TKO, TLP> {
    type Output;
    fn contract_first_third_fourth_with_first(
        self,
        object_a: TIM,
        object_b: TKO,
        object_c: TLP,
    ) -> Self::Output;
}

impl<const D: usize, I, J, K, L, M, O, P, U>
    ContractFirstThirdFourthWithFirst<
        &TensorRank2<D, I, M, U>,
        &TensorRank2<D, K, O, U>,
        &TensorRank2<D, L, P, U>,
    > for TensorRank4<D, I, J, K, L, U>
{
    type Output = TensorRank4<D, M, J, O, P, U>;
    fn contract_first_third_fourth_with_first(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M, U>,
        tensor_rank_2_b: &TensorRank2<D, K, O, U>,
        tensor_rank_2_c: &TensorRank2<D, L, P, U>,
    ) -> Self::Output {
        // One index at a time, for the same reason.
        let first = canonical_transform_first(self.canonical(), tensor_rank_2_a.canonical());
        let third = canonical_contract_third_with_first(&first, tensor_rank_2_b.canonical());
        relabel(canonical_transform_fourth(
            &third,
            tensor_rank_2_c.canonical(),
        ))
    }
}

/// Transforms the second index of a rank-4 tensor.
///
/// Contracts it with the first index of a rank-2 tensor.
pub trait ContractSecondWithFirst<TJN> {
    type Output;
    fn contract_second_with_first(self, tensor_rank_2: TJN) -> Self::Output;
}

impl<const D: usize, I, J, K, L, N, U> ContractSecondWithFirst<&TensorRank2<D, J, N, U>>
    for TensorRank4<D, I, J, K, L, U>
{
    type Output = TensorRank4<D, I, N, K, L, U>;
    fn contract_second_with_first(self, tensor_rank_2: &TensorRank2<D, J, N, U>) -> Self::Output {
        relabel(canonical_contract_second_with_first(
            self.canonical(),
            tensor_rank_2.canonical(),
        ))
    }
}

/// Contracts the second and fourth indices of a rank-4 tensor with two vectors.
///
/// Yields the rank-2 tensor over the remaining first and third indices.
pub trait ContractSecondFourthWithFirst<TJ, TL> {
    type Output;
    fn contract_second_fourth_with_first(&self, object_a: TJ, object_b: TL) -> Self::Output;
}

impl<const D: usize, I, J, K, L, U>
    ContractSecondFourthWithFirst<&TensorRank1<D, J, U>, &TensorRank1<D, L, U>>
    for TensorRank4<D, I, J, K, L, U>
{
    type Output = TensorRank2<D, I, K, U>;
    fn contract_second_fourth_with_first(
        &self,
        tensor_rank_1_a: &TensorRank1<D, J, U>,
        tensor_rank_1_b: &TensorRank1<D, L, U>,
    ) -> Self::Output {
        // The scaled copy of the second vector this used to make depended on
        // neither of the indices it was made inside, so it is a dot product and
        // a scalar multiply instead.
        let mut output = TensorRank2::zero();
        self.iter()
            .zip(output.iter_mut())
            .for_each(|(self_i, output_i)| {
                self_i.iter().zip(tensor_rank_1_a.iter()).for_each(
                    |(self_ij, tensor_rank_1_a_j)| {
                        self_ij
                            .iter()
                            .zip(output_i.iter_mut())
                            .for_each(|(self_ijk, output_ik)| {
                                *output_ik += (self_ijk * tensor_rank_1_b) * tensor_rank_1_a_j
                            })
                    },
                )
            });
        output
    }
}

/// Transforms the third index of a rank-4 tensor.
///
/// Contracts it with the first index of a rank-2 tensor.
pub trait ContractThirdWithFirst<TKL> {
    type Output;
    fn contract_third_with_first(&self, tensor: TKL) -> Self::Output;
}

impl<const D: usize, I, J, K, L, M, U> ContractThirdWithFirst<&TensorRank2<D, M, K, U>>
    for TensorRank4<D, I, J, M, L, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn contract_third_with_first(&self, tensor_rank_2: &TensorRank2<D, M, K, U>) -> Self::Output {
        relabel(canonical_contract_third_with_first(
            self.canonical(),
            tensor_rank_2.canonical(),
        ))
    }
}

/// Double-contracts the third and fourth indices of a rank-4 tensor.
///
/// Contracts against the leading two indices of another rank-2 or rank-4 tensor.
pub trait ContractThirdFourthWithFirstSecond<TKL> {
    type Output;
    fn contract_third_fourth_with_first_second(self, tensor: TKL) -> Self::Output;
}

impl<const D: usize, I, J, K, L, U, V> ContractThirdFourthWithFirstSecond<&TensorRank2<D, K, L, V>>
    for TensorRank4<D, I, J, K, L, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2<D, I, J, <U as UnitMul<V>>::Output>;
    fn contract_third_fourth_with_first_second(
        self,
        tensor_rank_2: &TensorRank2<D, K, L, V>,
    ) -> Self::Output {
        relabel_rank_2(canonical_contract_34_12_rank_2(
            self.into_canonical(),
            tensor_rank_2.canonical(),
        ))
    }
}

fn canonical_contract_34_12_rank_2<const D: usize>(
    tensor_rank_4: TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank2<D, Reference, Reference, Dimensionless> {
    tensor_rank_4
        .into_iter()
        .map(|self_i| {
            self_i
                .into_iter()
                .map(|self_ij| self_ij.full_contraction(tensor_rank_2))
                .collect()
        })
        .collect()
}

impl<const D: usize, I, J, K, L, M, N, U, V>
    ContractThirdFourthWithFirstSecond<&TensorRank4<D, K, L, M, N, V>>
    for TensorRank4<D, I, J, K, L, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank4<D, I, J, M, N, <U as UnitMul<V>>::Output>;
    fn contract_third_fourth_with_first_second(
        self,
        tensor: &TensorRank4<D, K, L, M, N, V>,
    ) -> Self::Output {
        relabel(canonical_contract_34_12_rank_4(
            self.into_canonical(),
            tensor.canonical(),
        ))
    }
}

fn canonical_contract_34_12_rank_4<const D: usize>(
    tensor_rank_4: TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_4
        .into_iter()
        .map(|self_i| {
            self_i
                .into_iter()
                .map(|self_ij| {
                    self_ij
                        .into_iter()
                        .zip(tensor.iter())
                        .map(|(self_ijk, tensor_k)| {
                            self_ijk
                                .into_iter()
                                .zip(tensor_k.iter())
                                .map(|(self_ijkl, tensor_kl)| tensor_kl * self_ijkl)
                                .sum::<TensorRank2<D, Reference, Reference, Dimensionless>>()
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Transforms the first and second indices of a rank-4 tensor.
///
/// Contracts each with the second index of a corresponding rank-2 tensor.
pub trait ContractFirstSecondWithSecond<TI, TJ> {
    type Output;
    fn contract_first_second_with_second(self, object_a: TI, object_b: TJ) -> Self::Output;
}

impl<const D: usize, I, J, K, L, M, N, U>
    ContractFirstSecondWithSecond<&TensorRank2<D, I, M, U>, &TensorRank2<D, J, N, U>>
    for TensorRank4<D, M, N, K, L, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn contract_first_second_with_second(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M, U>,
        tensor_rank_2_b: &TensorRank2<D, J, N, U>,
    ) -> Self::Output {
        let mut output = TensorRank4::zero();
        output
            .iter_mut()
            .zip(tensor_rank_2_a.iter())
            .for_each(|(output_i, tensor_rank_2_a_i)| {
                output_i.iter_mut().zip(tensor_rank_2_b.iter()).for_each(
                    |(output_ij, tensor_rank_2_b_j)| {
                        self.iter().zip(tensor_rank_2_a_i.iter()).for_each(
                            |(self_m, tensor_rank_2_a_im)| {
                                self_m.iter().zip(tensor_rank_2_b_j.iter()).for_each(
                                    |(self_mn, tensor_rank_2_b_jn)| {
                                        *output_ij +=
                                            self_mn * tensor_rank_2_a_im * tensor_rank_2_b_jn
                                    },
                                )
                            },
                        )
                    },
                )
            });
        output
    }
}

impl<const D: usize, I, J, K, L, U> Div<TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Div<TensorRank0> for &TensorRank4<D, I, J, K, L, U> {
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, K, L, U> Div<&TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K, L, U> DivAssign<TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, K, L, U> DivAssign<&TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, I, J, K, L, U> Mul<TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Mul<&TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K, L, U> MulAssign<TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, K, L, U> MulAssign<&TensorRank0> for TensorRank4<D, I, J, K, L, U> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, I, J, K, L, M, U, V> Mul<TensorRank2<D, L, M, V>>
    for TensorRank4<D, I, J, K, L, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank4<D, I, J, K, M, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: TensorRank2<D, L, M, V>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| self_ij * &tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, L, M, U, V> Mul<&TensorRank2<D, L, M, V>>
    for TensorRank4<D, I, J, K, L, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank4<D, I, J, K, M, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, L, M, V>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| self_ij * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, J, K, L, M, U> Mul<TensorRank4<D, M, J, K, L, U>> for TensorRank1<D, M, U> {
    type Output = TensorRank3<D, J, K, L, U>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel_rank_3(canonical_rank_1_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, J, K, L, M, U> Mul<&TensorRank4<D, M, J, K, L, U>> for TensorRank1<D, M, U> {
    type Output = TensorRank3<D, J, K, L, U>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel_rank_3(canonical_rank_1_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, J, K, L, M, U> Mul<TensorRank4<D, M, J, K, L, U>> for &TensorRank1<D, M, U> {
    type Output = TensorRank3<D, J, K, L, U>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel_rank_3(canonical_rank_1_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, J, K, L, M, U> Mul<&TensorRank4<D, M, J, K, L, U>> for &TensorRank1<D, M, U> {
    type Output = TensorRank3<D, J, K, L, U>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel_rank_3(canonical_rank_1_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, I, J, K, L, M, U> Mul<TensorRank4<D, M, J, K, L, U>>
    for TensorRank2<D, I, M, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel(canonical_rank_2_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, I, J, K, L, M, U> Mul<&TensorRank4<D, M, J, K, L, U>>
    for TensorRank2<D, I, M, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel(canonical_rank_2_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, I, J, K, L, M, U> Mul<TensorRank4<D, M, J, K, L, U>>
    for &TensorRank2<D, I, M, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel(canonical_rank_2_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, I, J, K, L, M, U> Mul<&TensorRank4<D, M, J, K, L, U>>
    for &TensorRank2<D, I, M, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L, U>) -> Self::Output {
        relabel(canonical_rank_2_times_rank_4(
            self.canonical(),
            tensor_rank_4.canonical(),
        ))
    }
}

impl<const D: usize, I, J, K, L, U> Add for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_4: Self) -> Self::Output {
        self += tensor_rank_4;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Add<&Self> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_4: &Self) -> Self::Output {
        self += tensor_rank_4;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Add<TensorRank4<D, I, J, K, L, U>>
    for &TensorRank4<D, I, J, K, L, U>
{
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn add(self, mut tensor_rank_4: TensorRank4<D, I, J, K, L, U>) -> Self::Output {
        tensor_rank_4 += self;
        tensor_rank_4
    }
}

impl<const D: usize, I, J, K, L, U> AddAssign for TensorRank4<D, I, J, K, L, U> {
    fn add_assign(&mut self, tensor_rank_4: Self) {
        self.canonical_mut()
            .add_assign_core(tensor_rank_4.into_canonical());
    }
}

impl<const D: usize, I, J, K, L, U> AddAssign<&Self> for TensorRank4<D, I, J, K, L, U> {
    fn add_assign(&mut self, tensor_rank_4: &Self) {
        self.canonical_mut()
            .add_assign_ref_core(tensor_rank_4.canonical());
    }
}

impl<const D: usize, I, J, K, L, U> Sub for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_4: Self) -> Self::Output {
        self -= tensor_rank_4;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Sub<&Self> for TensorRank4<D, I, J, K, L, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_4: &Self) -> Self::Output {
        self -= tensor_rank_4;
        self
    }
}

impl<const D: usize, I, J, K, L, U> Sub for &TensorRank4<D, I, J, K, L, U> {
    type Output = TensorRank4<D, I, J, K, L, U>;
    fn sub(self, tensor_rank_4: Self) -> Self::Output {
        tensor_rank_4
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_4_i, self_i)| self_i - tensor_rank_4_i)
            .collect()
    }
}

impl<const D: usize, I, J, K, L, U> SubAssign for TensorRank4<D, I, J, K, L, U> {
    fn sub_assign(&mut self, tensor_rank_4: Self) {
        self.canonical_mut()
            .sub_assign_core(tensor_rank_4.into_canonical());
    }
}

impl<const D: usize, I, J, K, L, U> SubAssign<&Self> for TensorRank4<D, I, J, K, L, U> {
    fn sub_assign(&mut self, tensor_rank_4: &Self) {
        self.canonical_mut()
            .sub_assign_ref_core(tensor_rank_4.canonical());
    }
}

// A quantity carries its unit into the tensor it scales.

impl<const D: usize, I, J, K, L, U, V> Mul<Quantity<V>> for TensorRank4<D, I, J, K, L, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank4<D, I, J, K, L, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() * quantity.value())
    }
}

impl<const D: usize, I, J, K, L, U, V> Div<Quantity<V>> for TensorRank4<D, I, J, K, L, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank4<D, I, J, K, L, <U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() / quantity.value())
    }
}

fn canonical_dyad_ij_kl<const D: usize>(
    tensor_rank_2_a: &TensorRank2<D, Reference, Reference, Dimensionless>,
    tensor_rank_2_b: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_2_a
        .iter()
        .map(|tensor_rank_2_a_i| {
            tensor_rank_2_a_i
                .iter()
                .map(|tensor_rank_2_a_ij| tensor_rank_2_b * tensor_rank_2_a_ij)
                .collect()
        })
        .collect()
}

fn canonical_dyad_ik_jl<const D: usize>(
    tensor_rank_2_a: &TensorRank2<D, Reference, Reference, Dimensionless>,
    tensor_rank_2_b: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_2_a
        .iter()
        .map(|tensor_rank_2_a_i| {
            tensor_rank_2_b
                .iter()
                .map(|tensor_rank_2_b_j| {
                    tensor_rank_2_a_i
                        .iter()
                        .map(|tensor_rank_2_a_ik| tensor_rank_2_b_j * tensor_rank_2_a_ik)
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn canonical_dyad_il_jk<const D: usize>(
    tensor_rank_2_a: &TensorRank2<D, Reference, Reference, Dimensionless>,
    tensor_rank_2_b: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_2_a
        .iter()
        .map(|tensor_rank_2_a_i| {
            tensor_rank_2_b
                .iter()
                .map(|tensor_rank_2_b_j| {
                    tensor_rank_2_b_j
                        .iter()
                        .map(|tensor_rank_2_b_jk| tensor_rank_2_a_i * tensor_rank_2_b_jk)
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn canonical_rank_1_times_rank_4<const D: usize>(
    tensor_rank_1: &TensorRank1<D, Reference, Dimensionless>,
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
) -> TensorRank3<D, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_1
        .iter()
        .zip(tensor_rank_4.iter())
        .map(|(&tensor_rank_1_m, tensor_rank_4_m)| tensor_rank_4_m * tensor_rank_1_m)
        .sum()
}

fn canonical_rank_2_times_rank_4<const D: usize>(
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    tensor_rank_2
        .iter()
        .map(|tensor_rank_2_i| tensor_rank_2_i * tensor_rank_4)
        .collect()
}

/// Contracts the second index against the first of a rank 2.
///
/// `out[i][j][k][l] = sum_s tensor_rank_4[i][s][k][l] * tensor_rank_2[s][j]`
///
/// Accumulated in place, so none of the scaled rank-2 blocks the sum would
/// otherwise build are materialized.
fn canonical_contract_second_with_first<const D: usize>(
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    let mut output = TensorRank4::zero();
    tensor_rank_4
        .iter()
        .zip(output.iter_mut())
        .for_each(|(tensor_rank_4_i, output_i)| {
            tensor_rank_4_i.iter().zip(tensor_rank_2.iter()).for_each(
                |(tensor_rank_4_is, tensor_rank_2_s)| {
                    tensor_rank_2_s.iter().zip(output_i.iter_mut()).for_each(
                        |(tensor_rank_2_sj, output_ij)| {
                            output_ij.iter_mut().zip(tensor_rank_4_is.iter()).for_each(
                                |(output_ijk, tensor_rank_4_isk)| {
                                    output_ijk
                                        .iter_mut()
                                        .zip(tensor_rank_4_isk.iter())
                                        .for_each(|(output_ijkl, tensor_rank_4_iskl)| {
                                            *output_ijkl += tensor_rank_4_iskl * tensor_rank_2_sj
                                        })
                                },
                            )
                        },
                    )
                },
            )
        });
    output
}

/// Contracts the third index against the first of a rank 2.
///
/// `out[i][j][k][l] = sum_m tensor_rank_2[m][k] * tensor_rank_4[i][j][m][l]`
///
/// Accumulated in place, so the outer product this formed for every `m` and
/// then summed is never built.
fn canonical_contract_third_with_first<const D: usize>(
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    let mut output = TensorRank4::zero();
    tensor_rank_4
        .iter()
        .zip(output.iter_mut())
        .for_each(|(tensor_rank_4_i, output_i)| {
            tensor_rank_4_i.iter().zip(output_i.iter_mut()).for_each(
                |(tensor_rank_4_ij, output_ij)| {
                    tensor_rank_4_ij.iter().zip(tensor_rank_2.iter()).for_each(
                        |(tensor_rank_4_ijm, tensor_rank_2_m)| {
                            tensor_rank_2_m.iter().zip(output_ij.iter_mut()).for_each(
                                |(tensor_rank_2_mk, output_ijk)| {
                                    output_ijk
                                        .iter_mut()
                                        .zip(tensor_rank_4_ijm.iter())
                                        .for_each(|(output_ijkl, tensor_rank_4_ijml)| {
                                            *output_ijkl += tensor_rank_2_mk * tensor_rank_4_ijml
                                        })
                                },
                            )
                        },
                    )
                },
            )
        });
    output
}

/// Transforms the first index, contracting it with the first of a rank 2.
///
/// `out[i][j][k][l] = sum_m tensor_rank_4[m][j][k][l] * tensor_rank_2[m][i]`
fn canonical_transform_first<const D: usize>(
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    let mut output = TensorRank4::zero();
    tensor_rank_4.iter().zip(tensor_rank_2.iter()).for_each(
        |(tensor_rank_4_m, tensor_rank_2_m)| {
            tensor_rank_2_m.iter().zip(output.iter_mut()).for_each(
                |(tensor_rank_2_mi, output_i)| {
                    output_i.iter_mut().zip(tensor_rank_4_m.iter()).for_each(
                        |(output_ij, tensor_rank_4_mj)| {
                            output_ij.iter_mut().zip(tensor_rank_4_mj.iter()).for_each(
                                |(output_ijk, tensor_rank_4_mjk)| {
                                    output_ijk
                                        .iter_mut()
                                        .zip(tensor_rank_4_mjk.iter())
                                        .for_each(|(output_ijkl, tensor_rank_4_mjkl)| {
                                            *output_ijkl += tensor_rank_4_mjkl * tensor_rank_2_mi
                                        })
                                },
                            )
                        },
                    )
                },
            )
        },
    );
    output
}

/// Transforms the fourth index, contracting it with the first of a rank 2.
///
/// `out[i][j][k][l] = sum_m tensor_rank_4[i][j][k][m] * tensor_rank_2[m][l]`
fn canonical_transform_fourth<const D: usize>(
    tensor_rank_4: &TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless>,
    tensor_rank_2: &TensorRank2<D, Reference, Reference, Dimensionless>,
) -> TensorRank4<D, Reference, Reference, Reference, Reference, Dimensionless> {
    let mut output = TensorRank4::zero();
    tensor_rank_4
        .iter()
        .zip(output.iter_mut())
        .for_each(|(tensor_rank_4_i, output_i)| {
            tensor_rank_4_i.iter().zip(output_i.iter_mut()).for_each(
                |(tensor_rank_4_ij, output_ij)| {
                    tensor_rank_4_ij.iter().zip(output_ij.iter_mut()).for_each(
                        |(tensor_rank_4_ijk, output_ijk)| {
                            tensor_rank_4_ijk.iter().zip(tensor_rank_2.iter()).for_each(
                                |(tensor_rank_4_ijkm, tensor_rank_2_m)| {
                                    output_ijk.iter_mut().zip(tensor_rank_2_m.iter()).for_each(
                                        |(output_ijkl, tensor_rank_2_ml)| {
                                            *output_ijkl += tensor_rank_4_ijkm * tensor_rank_2_ml
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            )
        });
    output
}
