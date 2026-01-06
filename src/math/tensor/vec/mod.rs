use crate::math::{Tensor, TensorRank0, TensorRank1, TensorVec};
use std::{
    collections::VecDeque,
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice, vec,
};

#[derive(Clone, Debug, PartialEq)]
pub struct TensorVector<T>(Vec<T>);
// where
//     T: Tensor;

// NEED TO MOVE SOMEWHERE ELSE

pub type TensorRank1RefVec<'a, const D: usize, const I: usize> =
    TensorVector<&'a TensorRank1<D, I>>;

impl<'a, const D: usize, const I: usize> TensorRank1RefVec<'a, D, I> {
    pub fn iter(&self) -> impl Iterator<Item = &&TensorRank1<D, I>> {
        self.0.iter()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, const D: usize, const I: usize> Index<usize> for TensorRank1RefVec<'a, D, I> {
    type Output = TensorRank1<D, I>;
    fn index(&self, index: usize) -> &Self::Output {
        self.0[index]
    }
}

// NEED TO MOVE SOMEWHERE ELSE

impl<T> TensorVector<T>
where
    T: Tensor,
{
    /// Returns a raw pointer to the vector’s buffer, or a dangling raw pointer valid for zero sized reads if the vector didn’t allocate.
    pub const fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
}

impl<T> Default for TensorVector<T>
where
    T: Tensor,
{
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T, const N: usize> From<[T; N]> for TensorVector<T>
where
    T: Tensor,
{
    fn from(array: [T; N]) -> Self {
        Self(array.to_vec())
    }
}

impl<T> From<&[T]> for TensorVector<T>
where
    T: Tensor,
{
    fn from(slice: &[T]) -> Self {
        Self(slice.to_vec())
    }
}

impl<T> From<Vec<T>> for TensorVector<T>
where
    T: Tensor,
{
    fn from(vec: Vec<T>) -> Self {
        Self(vec)
    }
}

impl<T> From<TensorVector<T>> for Vec<T>
where
    T: Tensor,
{
    fn from(tensor_vector: TensorVector<T>) -> Self {
        tensor_vector.0
    }
}

impl<T> From<VecDeque<T>> for TensorVector<T>
where
    T: Tensor,
{
    fn from(vec_deque: VecDeque<T>) -> Self {
        Self(vec_deque.into())
    }
}

impl<T> From<TensorVector<T>> for VecDeque<T>
where
    T: Tensor,
{
    fn from(tensor_vector: TensorVector<T>) -> Self {
        tensor_vector.0.into()
    }
}

impl<T> Display for TensorVector<T>
where
    T: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
        // write!(f, "\x1B[s")?;
        // write!(f, "[[")?;
        // self.iter().enumerate().try_for_each(|(i, tensor_rank_1)| {
        //     tensor_rank_1
        //         .iter()
        //         .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
        //     if i + 1 < W {
        //         writeln!(f, "\x1B[2D],")?;
        //         write!(f, "\x1B[u")?;
        //         write!(f, "\x1B[{}B [", i + 1)?;
        //     }
        //     Ok(())
        // })?;
        // write!(f, "\x1B[2D]]")
    }
}

impl<T> Extend<T> for TensorVector<T>
where
    T: Tensor,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.0.extend(iter)
    }
}

impl<T> Index<usize> for TensorVector<T>
where
    T: Tensor,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for TensorVector<T>
where
    T: Tensor,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> Tensor for TensorVector<T>
where
    T: Tensor,
{
    type Item = T;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn size(&self) -> usize {
        self.len() * self[0].size() // not a huge fan of this since T could be another Vec and each could have a different size
    }
}

impl<T> FromIterator<T> for TensorVector<T>
// where
//     T: Tensor,
{
    fn from_iter<Ii: IntoIterator<Item = T>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl<T> IntoIterator for TensorVector<T>
where
    T: Tensor,
{
    type Item = T;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a TensorVector<T>
where
    T: Tensor,
{
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> Sum for TensorVector<T>
where
    T: Tensor,
{
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

impl<T> TensorVec for TensorVector<T>
where
    T: Tensor,
{
    type Item = T;
    fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0)
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn new() -> Self {
        Self(Vec::new())
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
}

impl<T> Div<TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T> Div<&TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T> DivAssign<TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<T> DivAssign<&TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<T> Mul<TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T> Mul<&TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T> Mul<TensorRank0> for &TensorVector<T>
where
    T: Tensor,
    for<'a> &'a T: Mul<&'a TensorRank0, Output = T>,
{
    type Output = TensorVector<T>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_0).collect()
    }
}

// impl<T> Mul<&TensorRank0> for &TensorVector<T>
// where
//     T: Tensor,
//     for <'a> &'a T: Mul<&'a TensorRank0, Output=T>
// {
//     type Output = TensorVector<T>;
//     fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
//         self.iter().map(|self_i| self_i * tensor_rank_0).collect()
//     }
// }

impl<T> MulAssign<TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<T> MulAssign<&TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<T> Add for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_vec: Self) -> Self::Output {
        self += tensor_vec;
        self
    }
}

impl<T> Add<&Self> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_vec: &Self) -> Self::Output {
        self += tensor_vec;
        self
    }
}

impl<T> AddAssign for TensorVector<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_vec: Self) {
        self.iter_mut()
            .zip(tensor_vec)
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T> AddAssign<&Self> for TensorVector<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_vec: &Self) {
        self.iter_mut()
            .zip(tensor_vec.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T> Sub for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_vec: Self) -> Self::Output {
        self -= tensor_vec;
        self
    }
}

impl<T> Sub<&Self> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_vec: &Self) -> Self::Output {
        self -= tensor_vec;
        self
    }
}

impl<T> Sub for &TensorVector<T>
where
    T: Tensor,
    // for <'a> &'a T: Sub<&'a T, Output=T>
{
    type Output = TensorVector<T>;
    fn sub(self, _tensor_vec: Self) -> Self::Output {
        unimplemented!()
        // self
        //     .iter()
        //     .zip(tensor_vec.iter())
        //     .map(|(self_entry, entry)| {
        //         self_entry - entry
        //     })
        //     .collect()
    }
}

impl<T> SubAssign for TensorVector<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_vec: Self) {
        self.iter_mut()
            .zip(tensor_vec)
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}

impl<T> SubAssign<&Self> for TensorVector<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_vec: &Self) {
        self.iter_mut()
            .zip(tensor_vec.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}
