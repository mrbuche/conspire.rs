use crate::math::{Current, Reference};
#[cfg(test)]
mod test;

use crate::math::assert::FiniteDifference;

use std::{
    array::from_fn,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    Tensor, TensorArray,
    rank_0::TensorRank0,
    rank_2::{
        TensorRank2, get_identity_1010_parts_1, get_identity_1010_parts_2,
        get_identity_1010_parts_3, get_levi_civita_parts,
    },
};

/// Returns the rank-3 Levi-Civita symbol.
pub fn levi_civita<I, J, K>() -> TensorRank3<3, I, J, K> {
    TensorRank3::from([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
}

/// A *d*-dimensional tensor of rank 3.
///
/// `D` is the dimension, `I`, `J`, `K` are the configurations.
#[repr(transparent)]
pub struct TensorRank3<const D: usize, I, J, K>(
    [TensorRank2<D, J, K>; D],
    pub(super) PhantomData<I>,
);

impl<const D: usize, I, J, K> Clone for TensorRank3<D, I, J, K> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<const D: usize, I, J, K> Debug for TensorRank3<D, I, J, K> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const D: usize, I, J, K> PartialEq for TensorRank3<D, I, J, K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J, K> TensorRank3<D, I, J, K> {
    /// Views the tensor with its configurations discarded, so that arithmetic is
    /// compiled once per dimension rather than once per configuration.
    fn canonical(&self) -> &TensorRank3<D, Reference, Reference, Reference> {
        unsafe { &*(self as *const Self as *const TensorRank3<D, Reference, Reference, Reference>) }
    }
    fn canonical_mut(&mut self) -> &mut TensorRank3<D, Reference, Reference, Reference> {
        unsafe { &mut *(self as *mut Self as *mut TensorRank3<D, Reference, Reference, Reference>) }
    }
    fn into_canonical(self) -> TensorRank3<D, Reference, Reference, Reference> {
        unsafe {
            (&self as *const Self)
                .cast::<TensorRank3<D, Reference, Reference, Reference>>()
                .read()
        }
    }
}

fn relabel<const D: usize, I, J, K>(
    tensor: TensorRank3<D, Reference, Reference, Reference>,
) -> TensorRank3<D, I, J, K> {
    unsafe {
        (&tensor as *const TensorRank3<D, Reference, Reference, Reference>)
            .cast::<TensorRank3<D, I, J, K>>()
            .read()
    }
}

impl<const D: usize> TensorRank3<D, Reference, Reference, Reference> {
    fn as_array_core(&self) -> [[[TensorRank0; D]; D]; D] {
        let mut array = [[[0.0; D]; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_2, tensor_rank_2)| *entry_rank_2 = tensor_rank_2.as_array());
        array
    }
    fn zero_core() -> Self {
        Self(from_fn(|_| TensorRank2::zero()), PhantomData)
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

impl<const D: usize, I, J, K> Default for TensorRank3<D, I, J, K> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize, I, J, K> From<[[[TensorRank0; D]; D]; D]> for TensorRank3<D, I, J, K> {
    fn from(array: [[[TensorRank0; D]; D]; D]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}

/// The 3D Levi-Civita permutation tensor, configurations (1, 1, 1).
pub const LEVI_CIVITA: TensorRank3<3, Current, Current, Current> =
    TensorRank3(get_levi_civita_parts(), PhantomData);

pub(crate) const fn get_identity_1010_parts<I, J, K>() -> [TensorRank3<3, I, J, K>; 3] {
    [
        TensorRank3(get_identity_1010_parts_1(), PhantomData),
        TensorRank3(get_identity_1010_parts_2(), PhantomData),
        TensorRank3(get_identity_1010_parts_3(), PhantomData),
    ]
}

impl<const D: usize, I, J, K> Display for TensorRank3<D, I, J, K> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, entry)| write!(f, "{entry},\n\x1B[u\x1B[{}B\x1B[1D", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D + 1)
    }
}

impl<const D: usize, I, J, K> FiniteDifference for TensorRank3<D, I, J, K> {
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
                            .filter(|&(&self_ijk, &comparator_ijk)| {
                                (self_ijk / comparator_ijk - 1.0).abs() >= epsilon
                                    && (self_ijk.abs() >= epsilon
                                        || comparator_ijk.abs() >= epsilon)
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, I, J, K> Tensor for TensorRank3<D, I, J, K> {
    type Item = TensorRank2<D, J, K>;
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
        D * D * D
    }
}

impl<const D: usize, I, J, K> IntoIterator for TensorRank3<D, I, J, K> {
    type Item = TensorRank2<D, J, K>;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, I, J, K> TensorArray for TensorRank3<D, I, J, K> {
    type Array = [[[TensorRank0; D]; D]; D];
    type Item = TensorRank2<D, J, K>;
    fn as_array(&self) -> Self::Array {
        self.canonical().as_array_core()
    }
    fn identity() -> Self {
        panic!()
    }
    fn zero() -> Self {
        relabel(TensorRank3::zero_core())
    }
}

impl<const D: usize, I, J, K> FromIterator<TensorRank2<D, J, K>> for TensorRank3<D, I, J, K> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank2<D, J, K>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_3 = Self::zero();
        tensor_rank_3
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_3_i, value_i)| *tensor_rank_3_i = value_i);
        tensor_rank_3
    }
}

impl<const D: usize, I, J, K> Index<usize> for TensorRank3<D, I, J, K> {
    type Output = TensorRank2<D, J, K>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, I, J, K> IndexMut<usize> for TensorRank3<D, I, J, K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, I, J, K> Sum for TensorRank3<D, I, J, K> {
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

impl<'a, const D: usize, I, J, K> Sum<&'a Self> for TensorRank3<D, I, J, K> {
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

impl<const D: usize, I, J, K> Div<TensorRank0> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K> Div<TensorRank0> for &TensorRank3<D, I, J, K> {
    type Output = TensorRank3<D, I, J, K>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, K> Div<&TensorRank0> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K> DivAssign<TensorRank0> for TensorRank3<D, I, J, K> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, K> DivAssign<&TensorRank0> for TensorRank3<D, I, J, K> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, I, J, K> Mul<TensorRank0> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K> Mul<&TensorRank0> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, J, K> Mul<TensorRank0> for &TensorRank3<D, I, J, K> {
    type Output = TensorRank3<D, I, J, K>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, K> Mul<&TensorRank0> for &TensorRank3<D, I, J, K> {
    type Output = TensorRank3<D, I, J, K>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, J, K> MulAssign<TensorRank0> for TensorRank3<D, I, J, K> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, I, J, K> MulAssign<&TensorRank0> for TensorRank3<D, I, J, K> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, I, J, K> Add for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn add(mut self, tensor_rank_3: Self) -> Self::Output {
        self += tensor_rank_3;
        self
    }
}

impl<const D: usize, I, J, K> Add<&Self> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn add(mut self, tensor_rank_3: &Self) -> Self::Output {
        self += tensor_rank_3;
        self
    }
}

impl<const D: usize, I, J, K> Add<TensorRank3<D, I, J, K>> for &TensorRank3<D, I, J, K> {
    type Output = TensorRank3<D, I, J, K>;
    fn add(self, mut tensor_rank_3: TensorRank3<D, I, J, K>) -> Self::Output {
        tensor_rank_3 += self;
        tensor_rank_3
    }
}

impl<const D: usize, I, J, K> AddAssign for TensorRank3<D, I, J, K> {
    fn add_assign(&mut self, tensor_rank_3: Self) {
        self.canonical_mut()
            .add_assign_core(tensor_rank_3.into_canonical());
    }
}

impl<const D: usize, I, J, K> AddAssign<&Self> for TensorRank3<D, I, J, K> {
    fn add_assign(&mut self, tensor_rank_3: &Self) {
        self.canonical_mut()
            .add_assign_ref_core(tensor_rank_3.canonical());
    }
}

impl<const D: usize, I, J, K> Sub for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn sub(mut self, tensor_rank_3: Self) -> Self::Output {
        self -= tensor_rank_3;
        self
    }
}

impl<const D: usize, I, J, K> Sub<&Self> for TensorRank3<D, I, J, K> {
    type Output = Self;
    fn sub(mut self, tensor_rank_3: &Self) -> Self::Output {
        self -= tensor_rank_3;
        self
    }
}

impl<const D: usize, I, J, K> Sub for &TensorRank3<D, I, J, K> {
    type Output = TensorRank3<D, I, J, K>;
    fn sub(self, tensor_rank_3: Self) -> Self::Output {
        tensor_rank_3
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_3_i, self_i)| self_i - tensor_rank_3_i)
            .collect()
    }
}

impl<const D: usize, I, J, K> SubAssign for TensorRank3<D, I, J, K> {
    fn sub_assign(&mut self, tensor_rank_3: Self) {
        self.canonical_mut()
            .sub_assign_core(tensor_rank_3.into_canonical());
    }
}

impl<const D: usize, I, J, K> SubAssign<&Self> for TensorRank3<D, I, J, K> {
    fn sub_assign(&mut self, tensor_rank_3: &Self) {
        self.canonical_mut()
            .sub_assign_ref_core(tensor_rank_3.canonical());
    }
}
