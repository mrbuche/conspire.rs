#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Tensor, TensorError, TensorRank2, TensorTuple, TensorVec,
    integrate::{ButcherTableau, IntegrationError, Times},
};
use crate::units::Dimensionless;
use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

const RECONSTRUCT_FAILED: &str =
    "the field increment has no reconstruction (matrix exponential undefined)";

/// The geometry of one integrated state field: how an increment advances the state.
///
/// The increment is an element of the field's tangent space (its Lie algebra
/// for a group-valued field), carried in the same Rust type as [`Self::Point`].
pub trait IntegrableField {
    /// The state value this field carries.
    type Point: Tensor;
    /// Advances `base` by `increment`.
    fn reconstruct(base: &Self::Point, increment: &Self::Point)
    -> Result<Self::Point, TensorError>;
    /// The RKMK correction: maps a rate-scaled increment to an algebra increment
    /// at the accumulated algebra element `sigma`. Flat fields are the identity.
    fn dexpinv(_sigma: &Self::Point, increment: Self::Point) -> Self::Point {
        increment
    }
}

/// A state in a flat vector space: the increment simply adds.
pub struct Flat<T>(PhantomData<T>);

impl<T> IntegrableField for Flat<T>
where
    T: Clone + Tensor,
    for<'a> T: Add<&'a T, Output = T>,
{
    type Point = T;
    fn reconstruct(base: &T, increment: &T) -> Result<T, TensorError> {
        Ok(base.clone() + increment)
    }
}

/// A state on the unimodular group (`det = 1`): the increment acts by the matrix
/// exponential, `X_{n+1} = exp(increment) X_n`, which is unimodular whenever the
/// increment is trace-free.
pub struct Unimodular<I>(PhantomData<I>);

impl<I> IntegrableField for Unimodular<I>
where
    TensorRank2<3, I, I, Dimensionless>: Tensor,
    for<'a> TensorRank2<3, I, I, Dimensionless>:
        Mul<&'a TensorRank2<3, I, I, Dimensionless>, Output = TensorRank2<3, I, I, Dimensionless>>,
{
    type Point = TensorRank2<3, I, I, Dimensionless>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Point,
    ) -> Result<Self::Point, TensorError> {
        Ok(increment.expm()? * base)
    }
    fn dexpinv(sigma: &Self::Point, increment: Self::Point) -> Self::Point {
        sigma.dexpinv(&increment)
    }
}

/// A composite of two fields; its state is the matching [`TensorTuple`], and an
/// increment reconstructs component-wise. Nests right for three or more fields.
pub struct Product<H, T>(PhantomData<(H, T)>);

impl<H, T> IntegrableField for Product<H, T>
where
    H: IntegrableField,
    T: IntegrableField,
    TensorTuple<H::Point, T::Point>: Tensor,
{
    type Point = TensorTuple<H::Point, T::Point>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Point,
    ) -> Result<Self::Point, TensorError> {
        Ok(TensorTuple(
            H::reconstruct(&base.0, &increment.0)?,
            T::reconstruct(&base.1, &increment.1)?,
        ))
    }
    fn dexpinv(sigma: &Self::Point, increment: Self::Point) -> Self::Point {
        TensorTuple(
            H::dexpinv(&sigma.0, increment.0),
            T::dexpinv(&sigma.1, increment.1),
        )
    }
}

/// Explicit Euler for a single [`IntegrableField`], one step per interval of `time`.
///
/// ```math
/// \mathbf{x}_{n+1} = \mathrm{reconstruct}\!\left(\mathbf{x}_n,\ h\,\mathbf{f}(t_n, \mathbf{x}_n)\right)
/// ```
pub fn integrate_euler<Fld, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Point, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Fld::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Fld: IntegrableField,
    Fld::Point: Clone + Differentiate<T>,
    for<'a> &'a Derivative<Fld::Point, T>: Mul<Quantity<T>, Output = Fld::Point>,
    U: TensorVec<Item = Fld::Point>,
{
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    points.push(point.clone());
    times.push(time[0]);
    for step in time.windows(2) {
        let increment = &rate(step[0], &point)? * (step[1] - step[0]);
        point = Fld::reconstruct(&point, &increment)
            .map_err(|_| IntegrationError::from(RECONSTRUCT_FAILED.to_string()))?;
        points.push(point.clone());
        times.push(step[1]);
    }
    Ok((times, points))
}

/// Runge–Kutta–Munthe-Kaas: a fixed-step [`ButcherTableau`] run in the field's
/// Lie algebra, with the [`IntegrableField::dexpinv`] correction per stage and a
/// single [`IntegrableField::reconstruct`] per step. Reduces to the plain tableau
/// on a flat field.
pub fn integrate_rkmk<Fld, Tab, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Point, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Fld::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone + Differentiate<T>,
    for<'a> &'a Derivative<Fld::Point, T>: Mul<Quantity<T>, Output = Fld::Point>,
    U: TensorVec<Item = Fld::Point>,
{
    let reconstruct = |base: &Fld::Point, increment: &Fld::Point| {
        Fld::reconstruct(base, increment)
            .map_err(|_| IntegrationError::from(RECONSTRUCT_FAILED.to_string()))
    };
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    points.push(point.clone());
    times.push(time[0]);
    for step in time.windows(2) {
        let dt = step[1] - step[0];
        let mut slopes: Vec<Fld::Point> = Vec::with_capacity(Tab::STAGES);
        for i in 0..Tab::STAGES {
            let sigma = if i == 0 {
                None
            } else {
                let mut accumulated = slopes[0].clone() * Tab::A[i][0];
                for (j, slope) in slopes.iter().enumerate().take(i).skip(1) {
                    accumulated += slope.clone() * Tab::A[i][j];
                }
                Some(accumulated)
            };
            let stage_point = match &sigma {
                Some(sigma) => reconstruct(&point, sigma)?,
                None => point.clone(),
            };
            let increment = &rate(step[0] + Tab::C[i] * dt, &stage_point)? * dt;
            slopes.push(match &sigma {
                Some(sigma) => Fld::dexpinv(sigma, increment),
                None => increment,
            });
        }
        let mut combined = slopes[0].clone() * Tab::B[0];
        for (i, slope) in slopes.iter().enumerate().skip(1) {
            combined += slope.clone() * Tab::B[i];
        }
        point = reconstruct(&point, &combined)?;
        points.push(point.clone());
        times.push(step[1]);
    }
    Ok((times, points))
}
