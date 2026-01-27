#[cfg(test)]
mod test;

use super::{Scalar, Tensor, TensorVec, Vector, integrate::IntegrationError};
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
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    /// Solution interpolation.
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError>;
}

/// Solution interpolation schemes with internal variables.
pub trait InterpolateSolutionInternalVariables<Y, Z, U, V>
where
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    /// Solution interpolation with internal variables.
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        zp: &V,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
    ) -> Result<(U, U, V), IntegrationError>;
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
