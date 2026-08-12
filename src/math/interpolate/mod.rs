#[cfg(test)]
mod test;

use super::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec, Vector,
    integrate::{IntegrationError, Times},
    unit::Time,
};
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
pub trait InterpolateSolution<Y, U, V, T = Time>
where
    Y: Differentiate<T> + Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    /// Solution interpolation.
    #[allow(clippy::too_many_arguments)]
    fn interpolate(
        &self,
        time: &Times<T>,
        tp: &Times<T>,
        yp: &U,
        dydtp: &V,
        k_sol: &[V],
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
    ) -> Result<(U, V), IntegrationError>;
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
