#[cfg(test)]
use crate::math::test::ErrorTensor;

use crate::math::{
    Jacobian, Matrix, Solution, Tensor, TensorRank0, TensorRank1Vec, TensorRank2, TensorVec,
    write_tensor_rank_0,
};
use std::{
    fmt::{Display, Formatter, Result},
    ops::{
        Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, RangeFrom, RangeTo, Sub,
        SubAssign,
    },
    vec::IntoIter,
};

/// A vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector(Vec<TensorRank0>);

impl Vector {
    pub fn as_slice(&self) -> &[TensorRank0] {
        self.0.as_slice()
    }
    pub fn ones(len: usize) -> Self {
        Self(vec![1.0; len])
    }
}

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
                    .filter(|&(&entry_i, &comparator_entry_i)| {
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
                    .filter(|&(&entry_i, &comparator_entry_i)| {
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
                        .filter(|&(&entry_i, &comparator_entry_i)| {
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

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for Vector {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        tensor_rank_1_vec.into_iter().flatten().collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2<D, I, J>> for Vector {
    fn from(tensor_rank_2: TensorRank2<D, I, J>) -> Self {
        tensor_rank_2.into_iter().flatten().collect()
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

impl Index<RangeTo<usize>> for Vector {
    type Output = [TensorRank0];
    fn index(&self, indices: RangeTo<usize>) -> &Self::Output {
        &self.0[indices]
    }
}

impl Index<RangeFrom<usize>> for Vector {
    type Output = [TensorRank0];
    fn index(&self, indices: RangeFrom<usize>) -> &Self::Output {
        &self.0[indices]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Tensor for Vector {
    type Item = TensorRank0;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl Solution for Vector {
    fn decrement_from_chained(&mut self, other: &mut Self, vector: Vector) {
        self.iter_mut()
            .chain(other.iter_mut())
            .zip(vector)
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
}

impl Jacobian for Vector {
    fn fill_into(self, vector: &mut Vector) {
        self.into_iter()
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn fill_into_chained(self, other: Self, vector: &mut Self) {
        self.into_iter()
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(entry_i, vector_i)| *vector_i = entry_i)
    }
}

impl IntoIterator for Vector {
    type Item = TensorRank0;
    type IntoIter = IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl TensorVec for Vector {
    type Item = TensorRank0;
    type Slice<'a> = &'a [TensorRank0];
    fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice<'_>) -> Self {
        slice.iter().copied().collect()
    }
    fn push(&mut self, item: Self::Item) {
        self.0.push(item)
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
            .for_each(|(self_entry, tensor_rank_0)| *self_entry += tensor_rank_0);
    }
}

impl AddAssign<&Self> for Vector {
    fn add_assign(&mut self, vector: &Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_0)| *self_entry += tensor_rank_0);
    }
}

impl Mul for Vector {
    type Output = TensorRank0;
    fn mul(self, vector: Self) -> Self::Output {
        self.iter()
            .zip(vector.iter())
            .map(|(self_i, vector_i)| self_i * vector_i)
            .sum()
    }
}

impl Mul<&Self> for Vector {
    type Output = TensorRank0;
    fn mul(self, vector: &Self) -> Self::Output {
        self.iter()
            .zip(vector.iter())
            .map(|(self_i, vector_i)| self_i * vector_i)
            .sum()
    }
}

impl Mul for &Vector {
    type Output = TensorRank0;
    fn mul(self, vector: Self) -> Self::Output {
        self.iter()
            .zip(vector.iter())
            .map(|(self_i, vector_i)| self_i * vector_i)
            .sum()
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

impl Sub<Vector> for &Vector {
    type Output = Vector;
    fn sub(self, mut vector: Vector) -> Self::Output {
        vector
            .iter_mut()
            .zip(self.iter())
            .for_each(|(vector_i, self_i)| *vector_i = self_i - *vector_i);
        vector
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

impl SubAssign<&[TensorRank0]> for Vector {
    fn sub_assign(&mut self, slice: &[TensorRank0]) {
        self.iter_mut()
            .zip(slice.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl Mul<&Matrix> for &Vector {
    type Output = Vector;
    fn mul(self, matrix: &Matrix) -> Self::Output {
        let mut output = Vector::zero(matrix.width());
        self.iter()
            .zip(matrix.iter())
            .for_each(|(self_i, matrix_i)| {
                output
                    .iter_mut()
                    .zip(matrix_i.iter())
                    .for_each(|(output_j, matrix_ij)| *output_j += self_i * matrix_ij)
            });
        output
    }
}

impl<const D: usize, const I: usize> Mul<&TensorRank1Vec<D, I>> for &Vector {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, I>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .enumerate()
            .map(|(a, entry_a)| {
                entry_a
                    .iter()
                    .enumerate()
                    .map(|(i, entry_a_i)| self[D * a + i] * entry_a_i)
                    .sum::<TensorRank0>()
            })
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank2<D, I, J>> for &Vector {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_2: &TensorRank2<D, I, J>) -> Self::Output {
        tensor_rank_2
            .iter()
            .enumerate()
            .map(|(i, entry_i)| {
                entry_i
                    .iter()
                    .enumerate()
                    .map(|(j, entry_ij)| self[D * i + j] * entry_ij)
                    .sum::<TensorRank0>()
            })
            .sum()
    }
}
