#[cfg(test)]
mod test;

use super::{Tensor, TensorArray, TensorRank0, TensorVec, Vector};
use std::ops::{Mul, Sub};

/// Linear interpolation schemes.
pub struct LinearInterpolation {}

/// One-dimensional interpolation schemes.
pub trait Interpolate1D<F, T>
where
    F: TensorVec<Item = T>,
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
    U: TensorVec<Item = Y>,
{
    /// Solution interpolation.
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U;
}

impl<F, T> Interpolate1D<F, T> for LinearInterpolation
where
    F: TensorVec<Item = T>,
    T: Tensor,
{
    fn interpolate_1d(x: &Vector, xp: &Vector, fp: &F) -> F {
        let mut i = 0;
        x.iter()
            .map(|x_k| {
                i = xp.iter().position(|xp_i| xp_i > x_k).unwrap();
                (fp[i].clone() - &fp[i - 1]) / (xp[i] - xp[i - 1]) * (x_k - xp[i - 1]) + &fp[i - 1]
            })
            .collect()
    }
}
