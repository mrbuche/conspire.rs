#[cfg(test)]
mod test;

use crate::math::{
    Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank2, TensorVec, write_tensor_rank_0,
};
use std::{
    fmt::{Display, Formatter, Result},
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A sparse collection of *d*-dimensional tensors of rank 1.
///
/// `D` is the dimension, `I` is the configuration.
#[derive(Clone, Debug)]
pub struct TensorRank1Sparse<const D: usize, const I: usize>(Vec<(usize, usize, TensorRank0)>);

impl<const D: usize, const I: usize> Display for TensorRank1Sparse<D, I> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        // write!(f, "\x1B[s")?;
        // write!(f, "[[")?;
        // self.iter().enumerate().try_for_each(|(i, tensor_rank_1)| {
        //     tensor_rank_1
        //         .iter()
        //         .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
        //     if i + 1 < self.len() {
        //         writeln!(f, "\x1B[2D],")?;
        //         write!(f, "\x1B[u")?;
        //         write!(f, "\x1B[{}B [", i + 1)?;
        //     }
        //     Ok(())
        // })?;
        // write!(f, "\x1B[2D]]")
        todo!()
    }
}

// impl<const D: usize, const I: usize> TensorVec for TensorRank1Sparse<D, I> {
//     type Item = TensorRank1<D, I>;
//     type Slice<'a> = &'a [[TensorRank0; D]];
//     fn append(&mut self, other: &mut Self) {
//         // self.0.append(&mut other.0)
//         todo!()
//     }
//     fn is_empty(&self) -> bool {
//         // self.0.is_empty()
//         todo!()
//     }
//     fn len(&self) -> usize {
//         // self.0.len()
//         todo!()
//     }
//     fn new(slice: Self::Slice<'_>) -> Self {
//         // slice
//         //     .iter()
//         //     .map(|slice_entry| Self::Item::new(*slice_entry))
//         //     .collect()
//         todo!()
//     }
//     fn push(&mut self, item: Self::Item) {
//         // self.0.push(item)
//         todo!()
//     }
//     fn zero(len: usize) -> Self {
//         // (0..len).map(|_| super::zero()).collect()
//         todo!()
//     }
// }

// impl<const D: usize, const I: usize> Tensor for TensorRank1Sparse<D, I> {
//     type Item = (usize, usize, TensorRank0);
//     fn get_at(&self, indices: &[usize]) -> &TensorRank0 {
//         // &self[indices[0]][indices[1]]
//         todo!()
//     }
//     fn get_at_mut(&mut self, indices: &[usize]) -> &mut TensorRank0 {
//         // &mut self[indices[0]][indices[1]]
//         todo!()
//     }
//     fn iter(&self) -> impl Iterator<Item = &Self::Item> {
//         self.0.iter()
//     }
//     fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
//         self.0.iter_mut()
//     }
// }

impl<const D: usize, const I: usize> TensorRank1Sparse<D, I> {
    // type Item = (TensorRank0, [usize; 2]);
    pub fn iter(&self) -> impl Iterator<Item = &(usize, usize, TensorRank0)> {
        self.0.iter()
    }
    // fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
    //     // self.0.iter_mut()
    //     todo!()
    // }
}
