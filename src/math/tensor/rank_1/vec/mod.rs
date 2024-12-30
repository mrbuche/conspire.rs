#[cfg(test)]
use super::super::test::ErrorTensor;

use crate::math::{
    write_tensor_rank_0, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank2, TensorVec,
};
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A vector of *d*-dimensional tensors of rank 1.
///
/// `D` is the dimension, `I` is the configuration.
#[derive(Debug)]
pub struct TensorRank1Vec<const D: usize, const I: usize>(pub Vec<TensorRank1<D, I>>);

impl<const D: usize, const I: usize> Display for TensorRank1Vec<D, I> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "\x1B[s")?;
        write!(f, "[[")?;
        self.iter().enumerate().try_for_each(|(i, tensor_rank_1)| {
            tensor_rank_1
                .iter()
                .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
            if i + 1 < self.len() {
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
impl<const D: usize, const I: usize> ErrorTensor for TensorRank1Vec<D, I> {
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

impl<const D: usize> From<TensorRank1Vec<D, 0>> for TensorRank1Vec<D, 1> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, 0>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> FromIterator<TensorRank1<D, I>> for TensorRank1Vec<D, I> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank1<D, I>>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl<const D: usize, const I: usize> Index<usize> for TensorRank1Vec<D, I> {
    type Output = TensorRank1<D, I>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize> IndexMut<usize> for TensorRank1Vec<D, I> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize> TensorRank1Vec<D, I> {
    /// Returns the sum of the full dot product of each tensor in each vector.
    pub fn dot(&self, tensors: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensors.iter())
            .map(|(entry, tensor)| entry * tensor)
            .sum()
    }
}

impl<'a, const D: usize, const I: usize> TensorVec<'a> for TensorRank1Vec<D, I> {
    type Item = TensorRank1<D, I>;
    type Slice = &'a [[TensorRank0; D]];
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice) -> Self {
        slice
            .iter()
            .map(|slice_entry| Self::Item::new(*slice_entry))
            .collect()
    }
    fn zero(len: usize) -> Self {
        (0..len).map(|_| super::zero()).collect()
    }
}

impl<const D: usize, const I: usize> Tensor for TensorRank1Vec<D, I> {
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

impl<const D: usize, const I: usize> Div<TensorRank0> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize> Div<&TensorRank0> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize> DivAssign<TensorRank0> for TensorRank1Vec<D, I> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize> DivAssign<&TensorRank0> for TensorRank1Vec<D, I> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize> Mul<TensorRank0> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}
impl<const D: usize, const I: usize> Mul<&TensorRank0> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize> Mul<&TensorRank0> for &TensorRank1Vec<D, I> {
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize> MulAssign<TensorRank0> for TensorRank1Vec<D, I> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize> MulAssign<&TensorRank0> for TensorRank1Vec<D, I> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize> Add for TensorRank1Vec<D, I> {
    type Output = Self;
    fn add(mut self, tensor_rank_1_vec: Self) -> Self::Output {
        self += tensor_rank_1_vec;
        self
    }
}

impl<const D: usize, const I: usize> Add<&Self> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn add(mut self, tensor_rank_1_vec: &Self) -> Self::Output {
        self += tensor_rank_1_vec;
        self
    }
}

impl<const D: usize, const I: usize> AddAssign for TensorRank1Vec<D, I> {
    fn add_assign(&mut self, tensor_rank_1_vec: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_vec.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl<const D: usize, const I: usize> AddAssign<&Self> for TensorRank1Vec<D, I> {
    fn add_assign(&mut self, tensor_rank_1_vec: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_vec.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for TensorRank1Vec<D, I>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_vec.iter())
            .map(|(self_entry, tensor_rank_1_vec_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_vec_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for TensorRank1Vec<D, I>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_vec.iter())
            .map(|(self_entry, tensor_rank_1_vec_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_vec_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_vec.iter())
            .map(|(self_entry, tensor_rank_1_vec_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_vec_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1_vec.iter())
            .map(|(self_entry, tensor_rank_1_vec_entry)| {
                TensorRank2::dyad(self_entry, tensor_rank_1_vec_entry)
            })
            .sum()
    }
}

impl<const D: usize, const I: usize> Sub for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1_vec: Self) -> Self::Output {
        self -= tensor_rank_1_vec;
        self
    }
}

impl<const D: usize, const I: usize> Sub<&Self> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1_vec: &Self) -> Self::Output {
        self -= tensor_rank_1_vec;
        self
    }
}

impl<const D: usize, const I: usize> SubAssign for TensorRank1Vec<D, I> {
    fn sub_assign(&mut self, tensor_rank_1_vec: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_vec.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl<const D: usize, const I: usize> SubAssign<&Self> for TensorRank1Vec<D, I> {
    fn sub_assign(&mut self, tensor_rank_1_vec: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_vec.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}
