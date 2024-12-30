#[cfg(test)]
use crate::math::test::ErrorTensor;

use crate::math::{write_tensor_rank_0, Tensor, TensorRank0, TensorVec};
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A vector.
#[derive(Debug)]
pub struct Vector(Vec<TensorRank0>);

#[cfg(test)]
impl ErrorTensor for Vector {
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

impl Display for Vector {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "\x1B[s")?;
        write!(f, "[")?;
        self.0.chunks(5).enumerate().try_for_each(|(i, chunk)| {
            chunk
                .iter()
                .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
            if (i + 1) * 5 < self.len() {
                writeln!(f, "\x1B[2D,")?;
                write!(f, "\x1B[u")?;
                write!(f, "\x1B[{}B ", i + 1)?;
            }
            Ok(())
        })?;
        write!(f, "\x1B[2D]")?;
        Ok(())
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl FromIterator<TensorRank0> for Vector {
    fn from_iter<Ii: IntoIterator<Item = TensorRank0>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl Index<usize> for Vector {
    type Output = TensorRank0;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Tensor for Vector {
    type Item = TensorRank0;
    fn copy(&self) -> Self {
        self.iter().map(|entry| entry.copy()).collect()
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<'a> TensorVec<'a> for Vector {
    type Item = TensorRank0;
    type Slice = &'a [TensorRank0];
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice) -> Self {
        slice.iter().copied().collect()
    }
    fn zero(len: usize) -> Self {
        Self(vec![0.0; len])
    }
}

impl Div<TensorRank0> for Vector {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl Div<&TensorRank0> for Vector {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl DivAssign<TensorRank0> for Vector {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl DivAssign<&TensorRank0> for Vector {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl Mul<TensorRank0> for Vector {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}
impl Mul<&TensorRank0> for Vector {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl Mul<&TensorRank0> for &Vector {
    type Output = Vector;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl MulAssign<TensorRank0> for Vector {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl MulAssign<&TensorRank0> for Vector {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl Add for Vector {
    type Output = Self;
    fn add(mut self, vector: Self) -> Self::Output {
        self += vector;
        self
    }
}

impl Add<&Self> for Vector {
    type Output = Self;
    fn add(mut self, vector: &Self) -> Self::Output {
        self += vector;
        self
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, vector: Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl AddAssign<&Self> for Vector {
    fn add_assign(&mut self, vector: &Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(mut self, vector: Self) -> Self::Output {
        self -= vector;
        self
    }
}

impl Sub<&Self> for Vector {
    type Output = Self;
    fn sub(mut self, vector: &Self) -> Self::Output {
        self -= vector;
        self
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, vector: Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl SubAssign<&Self> for Vector {
    fn sub_assign(&mut self, vector: &Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}
