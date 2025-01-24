use super::{Tensor, TensorArray, TensorRank0, Vector};
use std::ops::{Index, Mul, Sub};

/// Linear interpolation schemes.
pub struct LinearInterpolation {}

/// One-dimensional interpolation schemes.
pub trait Interpolate1D<F, T>
where
    F: FromIterator<T> + Index<usize, Output = T>,
    T: Tensor,
{
    /// One-dimensional interpolation.
    fn interpolate_1d(x: &Vector, xp: &Vector, fp: &F) -> F;
}

/// Solution interpolation schemes.
pub trait InterpolateSolution<Y, U>
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: FromIterator<Y> + Index<usize, Output = Y> + Tensor<Item = Y>,
{
    /// Solution interpolation.
    fn interpolate(&self, t: &Vector, tp: &Vector, yp: &U, f: impl Fn(&TensorRank0, &Y) -> Y) -> U;
}

impl<F, T> Interpolate1D<F, T> for LinearInterpolation
where
    F: FromIterator<T> + Index<usize, Output = T>,
    T: Tensor,
{
    fn interpolate_1d(x: &Vector, xp: &Vector, fp: &F) -> F {
        let mut i = 0;
        x.iter()
            .map(|x_k| {
                i = xp.iter().position(|xp_i| xp_i > x_k).unwrap();
                (fp[i].copy() - &fp[i - 1]) / (xp[i] - xp[i - 1]) * (x_k - xp[i - 1]) + &fp[i - 1]
            })
            .collect()
    }
}
