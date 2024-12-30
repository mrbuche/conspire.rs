#[cfg(test)]
mod test;

#[cfg(test)]
use super::super::test::ErrorTensor;

use std::array::from_fn;
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::math::{Tensor, TensorArray, TensorRank2};

use super::{
    super::{super::write_tensor_rank_0, Convert},
    TensorRank0, TensorRank1,
};

/// A list of *d*-dimensional tensors of rank 1.
///
/// `D` is the dimension, `I` is the configuration, `W` is the list length.
#[derive(Debug)]
pub struct TensorRank1List<const D: usize, const I: usize, const W: usize>(
    pub [TensorRank1<D, I>; W],
);

impl<const D: usize, const I: usize, const W: usize> Display for TensorRank1List<D, I, W> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "\x1B[s")?;
        write!(f, "[[")?;
        self.iter().enumerate().try_for_each(|(i, tensor_rank_1)| {
            tensor_rank_1
                .iter()
                .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
            if i + 1 < W {
                writeln!(f, "\x1B[2D],")?;
                write!(f, "\x1B[u")?;
                write!(f, "\x1B[{}B [", i + 1)?;
            }
            Ok(())
        })?;
        write!(f, "\x1B[2D]]")
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const W: usize> ErrorTensor for TensorRank1List<D, I, W> {
    fn error(
        &self,
        comparator: &Self,
        tol_abs: &TensorRank0,
        tol_rel: &TensorRank0,
    ) -> Option<usize> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(entry, comparator_entry)| {
                entry
                    .iter()
                    .zip(comparator_entry.iter())
                    .filter(|(&entry_i, &comparator_entry_i)| {
                        &(entry_i - comparator_entry_i).abs() >= tol_abs
                            && &(entry_i / comparator_entry_i - 1.0).abs() >= tol_rel
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(entry, comparator_entry)| {
                entry
                    .iter()
                    .zip(comparator_entry.iter())
                    .filter(|(&entry_i, &comparator_entry_i)| {
                        &(entry_i / comparator_entry_i - 1.0).abs() >= epsilon
                            && (&entry_i.abs() >= epsilon || &comparator_entry_i.abs() >= epsilon)
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            let auxillary = self
                .iter()
                .zip(comparator.iter())
                .map(|(entry, comparator_entry)| {
                    entry
                        .iter()
                        .zip(comparator_entry.iter())
                        .filter(|(&entry_i, &comparator_entry_i)| {
                            &(entry_i / comparator_entry_i - 1.0).abs() >= epsilon
                                && &(entry_i - comparator_entry_i).abs() >= epsilon
                                && (&entry_i.abs() >= epsilon
                                    || &comparator_entry_i.abs() >= epsilon)
                        })
                        .count()
                })
                .sum::<usize>()
                > 0;
            Some((auxillary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const W: usize> TensorRank1List<D, I, W> {
    /// Returns the sum of the full dot product of each tensor in each list.
    pub fn dot(&self, tensors: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensors.iter())
            .map(|(entry, tensor)| entry * tensor)
            .sum()
    }
}

impl<const D: usize, const I: usize, const W: usize> Tensor for TensorRank1List<D, I, W> {
    type Item = TensorRank1<D, I>;
    fn copy(&self) -> Self {
        self.iter().map(|entry| entry.copy()).collect()
    }
    fn get_at(&self, indices: &[usize]) -> &TensorRank0 {
        &self[indices[0]][indices[1]]
    }
    fn get_at_mut(&mut self, indices: &[usize]) -> &mut TensorRank0 {
        &mut self[indices[0]][indices[1]]
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const D: usize, const I: usize, const W: usize> TensorArray for TensorRank1List<D, I, W> {
    type Array = [[TensorRank0; D]; W];
    type Item = TensorRank1<D, I>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[0.0; D]; W];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry, tensor_rank_1)| *entry = tensor_rank_1.as_array());
        array
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn new(array: Self::Array) -> Self {
        array
            .iter()
            .map(|array_i| TensorRank1::new(*array_i))
            .collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| super::zero()))
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize>
    Convert<TensorRank1List<D, J, W>> for TensorRank1List<D, I, W>
{
    fn convert(&self) -> TensorRank1List<D, J, W> {
        self.iter()
            .map(|self_entry| self_entry.iter().copied().collect())
            .collect()
    }
}

impl<const D: usize, const W: usize> From<TensorRank1List<D, 0, W>> for TensorRank1List<D, 1, W> {
    fn from(tensor_rank_1_list: TensorRank1List<D, 0, W>) -> Self {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize, const W: usize> FromIterator<TensorRank1<D, I>>
    for TensorRank1List<D, I, W>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank1<D, I>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_1_list = Self::zero();
        tensor_rank_1_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_1_list_entry, entry)| *tensor_rank_1_list_entry = entry);
        tensor_rank_1_list
    }
}

impl<const D: usize, const I: usize, const W: usize> Index<usize> for TensorRank1List<D, I, W> {
    type Output = TensorRank1<D, I>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const W: usize> IndexMut<usize> for TensorRank1List<D, I, W> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const W: usize> std::iter::Sum for TensorRank1List<D, I, W> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        let mut output = TensorRank1List::zero();
        iter.for_each(|item| output += item);
        output
    }
}

impl<const D: usize, const I: usize, const W: usize> Div<TensorRank0> for TensorRank1List<D, I, W> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> Div<&TensorRank0>
    for TensorRank1List<D, I, W>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> DivAssign<TensorRank0>
    for TensorRank1List<D, I, W>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize> DivAssign<&TensorRank0>
    for TensorRank1List<D, I, W>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize> Mul<TensorRank0> for TensorRank1List<D, I, W> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}
impl<const D: usize, const I: usize, const W: usize> Mul<&TensorRank0>
    for TensorRank1List<D, I, W>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> Mul<&TensorRank0>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const W: usize> MulAssign<TensorRank0>
    for TensorRank1List<D, I, W>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize> MulAssign<&TensorRank0>
    for TensorRank1List<D, I, W>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize> Add for TensorRank1List<D, I, W> {
    type Output = Self;
    fn add(mut self, tensor_rank_1_list: Self) -> Self::Output {
        self += tensor_rank_1_list;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> Add<&Self> for TensorRank1List<D, I, W> {
    type Output = Self;
    fn add(mut self, tensor_rank_1_list: &Self) -> Self::Output {
        self += tensor_rank_1_list;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> AddAssign for TensorRank1List<D, I, W> {
    fn add_assign(&mut self, tensor_rank_1_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl<const D: usize, const I: usize, const W: usize> AddAssign<&Self> for TensorRank1List<D, I, W> {
    fn add_assign(&mut self, tensor_rank_1_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, tensor_rank_1_list_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_list_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, tensor_rank_1_list_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_list_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, tensor_rank_1_list_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_list_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for &TensorRank1List<D, I, W>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_list.iter())
            .map(|(self_entry, tensor_rank_1_list_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_list_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const W: usize> Sub for TensorRank1List<D, I, W> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1_list: Self) -> Self::Output {
        self -= tensor_rank_1_list;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> Sub<&Self> for TensorRank1List<D, I, W> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1_list: &Self) -> Self::Output {
        self -= tensor_rank_1_list;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize> SubAssign for TensorRank1List<D, I, W> {
    fn sub_assign(&mut self, tensor_rank_1_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl<const D: usize, const I: usize, const W: usize> SubAssign<&Self> for TensorRank1List<D, I, W> {
    fn sub_assign(&mut self, tensor_rank_1_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}
