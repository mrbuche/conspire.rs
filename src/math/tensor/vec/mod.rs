use crate::math::{Tensor, TensorRank0, TensorVec};
use std::{
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct TensorVector<T>(Vec<T>)
where
    T: Tensor;

impl<T> Default for TensorVector<T>
where
    T: Tensor,
{
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> From<Vec<T>> for TensorVector<T>
where
    T: Tensor,
{
    fn from(tensor_vec: Vec<T>) -> Self {
        Self(tensor_vec)
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
}

impl<T> FromIterator<T> for TensorVector<T>
where
    T: Tensor,
{
    fn from_iter<Ii: IntoIterator<Item = T>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
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
    // type Slice<'a> = &'a [[TensorRank0; D]];
    type Slice<'a> = &'a f32;
    fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0)
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice<'_>) -> Self {
        todo!("Do you need this method after all? Can you use From<> instead?")
        // slice
        //     .iter()
        //     .map(|slice_entry| Self::Item::new(*slice_entry))
        //     .collect()
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
        todo!("Do you need this method after all?")
        // (0..len).map(|_| super::zero()).collect()
    }
}

impl<T> Div<TensorRank0> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
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
        self *= &tensor_rank_0;
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
    fn add(mut self, tensor_list: Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<T> Add<&Self> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_list: &Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<T> AddAssign for TensorVector<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T> AddAssign<&Self> for TensorVector<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T> Sub for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<T> Sub<&Self> for TensorVector<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: &Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<T> SubAssign for TensorVector<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}

impl<T> SubAssign<&Self> for TensorVector<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}
