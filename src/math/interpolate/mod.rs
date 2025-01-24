use super::{Tensor, Vector};
use std::ops::Index;

/// Linear interpolation schemes.
pub struct LinearInterpolation {}

/// One-dimensional interpolation schemes.
pub trait Interpolate1D<F, T>
where
    F: FromIterator<T> + Index<usize, Output = T>,
    T: Tensor,
{
    fn interpolate_1d(x: &Vector, xp: &Vector, fp: &F) -> F;
}

// ode45 allegedly uses ntrp45.m
// so maybe you need to use interpolation schemes of the same order!

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
