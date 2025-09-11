#[cfg(test)]
use super::super::test::ErrorTensor;

use crate::math::{
    Jacobian, Solution, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank2,
    TensorRank2Vec2D, TensorVec, Vector, write_tensor_rank_0,
};
use std::{
    fmt::{self, Display, Formatter},
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A vector of *d*-dimensional tensors of rank 1.
///
/// `D` is the dimension, `I` is the configuration.
#[derive(Clone, Debug)]
pub struct TensorRank1Vec<const D: usize, const I: usize>(Vec<TensorRank1<D, I>>);

impl<const D: usize, const I: usize> Display for TensorRank1Vec<D, I> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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
            let auxiliary = self
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
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize> From<Vec<[TensorRank0; D]>> for TensorRank1Vec<D, I> {
    fn from(vec: Vec<[TensorRank0; D]>) -> Self {
        vec.into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> From<Vec<Vec<TensorRank0>>> for TensorRank1Vec<D, I> {
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        vec.into_iter()
            .map(|tensor_rank_1| tensor_rank_1.into())
            .collect()
    }
}

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for Vec<[TensorRank0; D]> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.clone().into())
            .collect()
    }
}

impl<const D: usize, const I: usize> From<TensorRank1Vec<D, I>> for Vec<Vec<TensorRank0>> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<D, I>) -> Self {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| tensor_rank_1.clone().into())
            .collect()
    }
}

impl From<TensorRank1Vec<3, 0>> for TensorRank1Vec<3, 1> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<3, 0>) -> Self {
        unsafe { transmute::<TensorRank1Vec<3, 0>, TensorRank1Vec<3, 1>>(tensor_rank_1_vec) }
    }
}

impl From<TensorRank1Vec<3, 1>> for TensorRank1Vec<3, 0> {
    fn from(tensor_rank_1_vec: TensorRank1Vec<3, 1>) -> Self {
        unsafe { transmute::<TensorRank1Vec<3, 1>, TensorRank1Vec<3, 0>>(tensor_rank_1_vec) }
    }
}

impl<const D: usize, const I: usize> From<Vector> for TensorRank1Vec<D, I> {
    fn from(vector: Vector) -> Self {
        let n = vector.len();
        if n % D != 0 {
            panic!("Vector length mismatch.")
        } else {
            (0..n / D)
                .map(|a| (0..D).map(|i| vector[D * a + i]).collect())
                .collect()
        }
    }
}

impl<const D: usize, const I: usize> From<&Vector> for TensorRank1Vec<D, I> {
    fn from(vector: &Vector) -> Self {
        let n = vector.len();
        if n % D != 0 {
            panic!("Vector length mismatch.")
        } else {
            (0..n / D)
                .map(|a| (0..D).map(|i| vector[D * a + i]).collect())
                .collect()
        }
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

impl<const D: usize, const I: usize> TensorVec for TensorRank1Vec<D, I> {
    type Item = TensorRank1<D, I>;
    type Slice<'a> = &'a [[TensorRank0; D]];
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
        slice
            .iter()
            .map(|slice_entry| Self::Item::new(*slice_entry))
            .collect()
    }
    fn push(&mut self, item: Self::Item) {
        self.0.push(item)
    }
    fn remove(&mut self, index: usize) -> Self::Item {
        self.0.remove(index)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&Self::Item) -> bool,
    {
        self.0.retain(f)
    }
    fn swap_remove(&mut self, index: usize) -> Self::Item {
        self.0.swap_remove(index)
    }
    fn zero(len: usize) -> Self {
        (0..len).map(|_| super::zero()).collect()
    }
}

impl<const D: usize, const I: usize> Tensor for TensorRank1Vec<D, I> {
    type Item = TensorRank1<D, I>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn norm_inf(&self) -> TensorRank0 {
        self.iter()
            .map(|tensor_rank_1| {
                tensor_rank_1
                    .iter()
                    .fold(0.0, |acc, entry| entry.abs().max(acc))
            })
            .reduce(TensorRank0::max)
            .unwrap()
    }
    fn num_entries(&self) -> usize {
        D * self.len()
    }
}

impl<const D: usize, const I: usize> IntoIterator for TensorRank1Vec<D, I> {
    type Item = TensorRank1<D, I>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, const I: usize> Solution for TensorRank1Vec<D, I> {
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .chain(other.iter_mut())
            .zip(vector)
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
}

impl<const D: usize, const I: usize> Jacobian for TensorRank1Vec<D, I> {
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

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank2Vec2D<D, I, J>>
    for &TensorRank1Vec<D, I>
{
    type Output = TensorRank1Vec<D, J>;
    fn div(self, _tensor_rank_2_vec_2d: TensorRank2Vec2D<D, I, J>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const I: usize> Sub<Vector> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(a, self_a)| {
            self_a
                .iter_mut()
                .enumerate()
                .for_each(|(i, self_a_i)| *self_a_i -= vector[D * a + i])
        });
        self
    }
}

impl<const D: usize, const I: usize> Sub<&Vector> for TensorRank1Vec<D, I> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(a, self_a)| {
            self_a
                .iter_mut()
                .enumerate()
                .for_each(|(i, self_a_i)| *self_a_i -= vector[D * a + i])
        });
        self
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

impl<const D: usize, const I: usize> Mul<TensorRank0> for &TensorRank1Vec<D, I> {
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
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

impl<const D: usize, const I: usize> Sub for &TensorRank1Vec<D, I> {
    type Output = TensorRank1Vec<D, I>;
    fn sub(self, tensor_rank_1_vec: Self) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_1_vec_a, self_a)| {
                tensor_rank_1_vec_a
                    .iter()
                    .zip(self_a.iter())
                    .map(|(tensor_rank_1_vec_a_i, self_a_i)| self_a_i - *tensor_rank_1_vec_a_i)
                    .collect()
            })
            .collect()
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
