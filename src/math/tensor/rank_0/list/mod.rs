#[cfg(test)]
mod test;

use super::{super::super::write_tensor_rank_0, Tensor, TensorArray, TensorRank0};
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A list of tensors of rank 0 (a list of scalars).
///
/// `W` is the list length.
#[derive(Debug)]
pub struct TensorRank0List<const W: usize>(pub [TensorRank0; W]);

/// Display implementation for rank-0 lists.
impl<const W: usize> Display for TensorRank0List<W> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "\x1B[s")?;
        write!(f, "[")?;
        self.0.chunks(5).enumerate().try_for_each(|(i, chunk)| {
            chunk
                .iter()
                .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
            if (i + 1) * 5 < W {
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

impl<const W: usize> Tensor for TensorRank0List<W> {
    type Item = TensorRank0;
    fn copy(&self) -> Self {
        self.iter().map(|entry| entry.copy()).collect()
    }
    fn full_contraction(&self, tensor_rank_0_list: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, tensor_rank_0)| self_entry * tensor_rank_0)
            .sum()
    }
    fn iter(&self) -> impl Iterator<Item = &TensorRank0> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const W: usize> TensorArray for TensorRank0List<W> {
    type Array = [TensorRank0; W];
    type Item = TensorRank0;
    fn as_array(&self) -> Self::Array {
        self.0
    }
    fn identity() -> Self {
        Self([1.0; W])
    }
    fn new(array: Self::Array) -> Self {
        Self(array)
    }
    fn zero() -> Self {
        Self([0.0; W])
    }
}

impl<const W: usize> FromIterator<TensorRank0> for TensorRank0List<W> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank0>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_0_list = Self::zero();
        tensor_rank_0_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_0, entry)| *tensor_rank_0 = entry);
        tensor_rank_0_list
    }
}

impl<const W: usize> Index<usize> for TensorRank0List<W> {
    type Output = TensorRank0;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const W: usize> IndexMut<usize> for TensorRank0List<W> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const W: usize> Add for TensorRank0List<W> {
    type Output = Self;
    fn add(mut self, tensor_rank_0_list: Self) -> Self::Output {
        self += tensor_rank_0_list;
        self
    }
}

impl<const W: usize> Add<&Self> for TensorRank0List<W> {
    type Output = Self;
    fn add(mut self, tensor_rank_0_list: &Self) -> Self::Output {
        self += tensor_rank_0_list;
        self
    }
}

impl<const W: usize> AddAssign for TensorRank0List<W> {
    fn add_assign(&mut self, tensor_rank_0_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_0_list.iter())
            .for_each(|(self_entry, tensor_rank_0)| *self_entry += tensor_rank_0);
    }
}

impl<const W: usize> AddAssign<&Self> for TensorRank0List<W> {
    fn add_assign(&mut self, tensor_rank_0_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_0_list.iter())
            .for_each(|(self_entry, tensor_rank_0)| *self_entry += tensor_rank_0);
    }
}

impl<const W: usize> Div<TensorRank0> for TensorRank0List<W> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const W: usize> Div<&TensorRank0> for TensorRank0List<W> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const W: usize> DivAssign<TensorRank0> for TensorRank0List<W> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const W: usize> DivAssign<&TensorRank0> for TensorRank0List<W> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const W: usize> Mul<TensorRank0> for TensorRank0List<W> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

// impl<const W: usize> Mul<TensorRank0> for &TensorRank0List<W> {
//     type Output = TensorRank0List<W>;
//     fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
//         self.iter().map(|self_i| self_i * tensor_rank_0).collect()
//     }
// }

impl<const W: usize> Mul<&TensorRank0> for TensorRank0List<W> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const W: usize> Mul<&TensorRank0> for &TensorRank0List<W> {
    type Output = TensorRank0List<W>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const W: usize> MulAssign<TensorRank0> for TensorRank0List<W> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const W: usize> MulAssign<&TensorRank0> for TensorRank0List<W> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const W: usize> Mul for TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, tensor_rank_0_list_entry)| self_entry * tensor_rank_0_list_entry)
            .sum()
    }
}

impl<const W: usize> Mul<&Self> for TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: &Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, tensor_rank_0_list_entry)| self_entry * tensor_rank_0_list_entry)
            .sum()
    }
}

impl<const W: usize> Mul<TensorRank0List<W>> for &TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: TensorRank0List<W>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, tensor_rank_0_list_entry)| self_entry * tensor_rank_0_list_entry)
            .sum()
    }
}

impl<const W: usize> Mul for &TensorRank0List<W> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_0_list: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_0_list.iter())
            .map(|(self_entry, tensor_rank_0_list_entry)| self_entry * tensor_rank_0_list_entry)
            .sum()
    }
}

impl<const W: usize> Sub for TensorRank0List<W> {
    type Output = Self;
    fn sub(mut self, tensor_rank_0_list: Self) -> Self::Output {
        self -= tensor_rank_0_list;
        self
    }
}

impl<const W: usize> Sub<&Self> for TensorRank0List<W> {
    type Output = Self;
    fn sub(mut self, tensor_rank_0_list: &Self) -> Self::Output {
        self -= tensor_rank_0_list;
        self
    }
}

impl<const W: usize> SubAssign for TensorRank0List<W> {
    fn sub_assign(&mut self, tensor_rank_0_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_0_list.iter())
            .for_each(|(self_entry, tensor_rank_0)| *self_entry -= tensor_rank_0);
    }
}

impl<const W: usize> SubAssign<&Self> for TensorRank0List<W> {
    fn sub_assign(&mut self, tensor_rank_0_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_0_list.iter())
            .for_each(|(self_entry, tensor_rank_0)| *self_entry -= tensor_rank_0);
    }
}
