#[cfg(test)]
mod test;

use crate::math::Scalar;
use std::ops::Add;

pub trait Cell: Add<Output = Self> + Copy + PartialOrd + Sized {
    const ONE: Self;
    const ZERO: Self;
    fn cells(self) -> usize;
    fn length(cells: usize) -> Option<Self>;
    fn scalar(self) -> Scalar;
    fn split(self) -> Self;
}

macro_rules! cell {
    ($($cell: ty), +) => {
        $(impl Cell for $cell {
            const ONE: Self = 1;
            const ZERO: Self = 0;
            fn cells(self) -> usize {
                self as usize
            }
            fn length(cells: usize) -> Option<Self> {
                (cells <= Self::MAX as usize).then_some(cells as Self)
            }
            fn scalar(self) -> Scalar {
                self as Scalar
            }
            fn split(self) -> Self {
                self / 2
            }
        })+
    }
}

cell!(u8, u16, u32, u64, usize);
