#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Norm, Quantity, Scalar, Tensor, TensorError, TensorRank2,
    TensorTuple, TensorVec,
    integrate::{ButcherTableau, EmbeddedTableau, IntegrationError, Times},
};
use crate::units::{Dimensionless, Time};
use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

const RECONSTRUCT_FAILED: &str =
    "the field increment has no reconstruction (matrix exponential undefined)";

/// The geometry of one integrated state field: how an increment advances the state.
///
/// [`Self::Increment`] is an element of the field's tangent space (its Lie algebra
/// for a group-valued field). It equals [`Self::Point`] for a flat field, but not
/// in general — e.g. `F_p` is a `Reference → Intermediate` map while its algebra
/// element `D_p Δt` maps `Intermediate → Intermediate`.
pub trait IntegrableField {
    /// The state value this field carries.
    type Point: Tensor;
    /// The tangent/algebra element that advances a [`Self::Point`].
    type Increment: Tensor;
    /// Advances `base` by `increment`.
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError>;
    /// The RKMK correction: maps a rate-scaled increment to an algebra increment
    /// at the accumulated algebra element `sigma`. Flat fields are the identity.
    fn dexpinv(_sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
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
    type Increment = T;
    fn reconstruct(base: &T, increment: &T) -> Result<T, TensorError> {
        Ok(base.clone() + increment)
    }
}

/// A state acted on by the matrix exponential, `X_{n+1} = exp(increment) X_n`,
/// staying on the unimodular group (`det = 1`) whenever the increment is
/// trace-free. The state maps `B → A` while its algebra element maps `A → A`,
/// so `F_p` (`Reference → Intermediate`) is `Unimodular<Intermediate, Reference>`.
pub struct Unimodular<A, B = A>(PhantomData<(A, B)>);

impl<A, B> IntegrableField for Unimodular<A, B>
where
    TensorRank2<3, A, B, Dimensionless>: Tensor,
    TensorRank2<3, A, A, Dimensionless>: Tensor,
    for<'a> TensorRank2<3, A, A, Dimensionless>:
        Mul<&'a TensorRank2<3, A, B, Dimensionless>, Output = TensorRank2<3, A, B, Dimensionless>>,
{
    type Point = TensorRank2<3, A, B, Dimensionless>;
    type Increment = TensorRank2<3, A, A, Dimensionless>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError> {
        Ok(increment.expm()? * base)
    }
    fn dexpinv(sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
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
    TensorTuple<H::Increment, T::Increment>: Tensor,
{
    type Point = TensorTuple<H::Point, T::Point>;
    type Increment = TensorTuple<H::Increment, T::Increment>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError> {
        Ok(TensorTuple(
            H::reconstruct(&base.0, &increment.0)?,
            T::reconstruct(&base.1, &increment.1)?,
        ))
    }
    fn dexpinv(sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
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
    mut rate: impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Fld::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Fld: IntegrableField,
    Fld::Point: Clone,
    Fld::Increment: Differentiate<T>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
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

fn reconstruct_or_err<Fld: IntegrableField>(
    base: &Fld::Point,
    increment: &Fld::Increment,
) -> Result<Fld::Point, IntegrationError> {
    Fld::reconstruct(base, increment)
        .map_err(|_| IntegrationError::from(RECONSTRUCT_FAILED.to_string()))
}

/// Fills `slopes` with one RKMK step's corrected stage slopes `k̃ᵢ` in the
/// field's Lie algebra: per stage combine the earlier `k̃ⱼ` by row `Aᵢ`,
/// `reconstruct` the stage point, evaluate the rate, scale by `dt`, apply
/// [`IntegrableField::dexpinv`] at the accumulated algebra element. `slopes` is
/// cleared first and reused, so a caller that steps in a loop allocates nothing.
/// The caller weights the entries by `B` (the step) and, for an embedded pair,
/// by `D` (the error estimate).
///
/// FSAL: `first_rate` seeds stage 0 (`C[0] == 0`) with a rate carried from the
/// previous step, skipping that evaluation; when `Tab::FSAL`, the raw rate at
/// the final stage (whose point is the step solution) is returned for the next
/// step to seed with.
fn rkmk_stage_slopes_into<Fld, Tab, T>(
    rate: &mut impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Increment, T>, String>,
    point: &Fld::Point,
    t: Quantity<T>,
    dt: Quantity<T>,
    slopes: &mut Vec<Fld::Increment>,
    first_rate: Option<&Derivative<Fld::Increment, T>>,
) -> Result<Option<Derivative<Fld::Increment, T>>, IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
{
    slopes.clear();
    slopes.reserve(Tab::STAGES);
    let mut carry = None;
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
            Some(sigma) => reconstruct_or_err::<Fld>(point, sigma)?,
            None => point.clone(),
        };
        let increment = match (i, first_rate) {
            (0, Some(seed)) => seed * dt,
            _ => {
                let raw = rate(t + dt * Tab::C[i], &stage_point)?;
                let increment = &raw * dt;
                if Tab::FSAL && i + 1 == Tab::STAGES {
                    carry = Some(raw);
                }
                increment
            }
        };
        slopes.push(match &sigma {
            Some(sigma) => Fld::dexpinv(sigma, increment),
            None => increment,
        });
    }
    Ok(carry)
}

fn weight<P>(slopes: &[P], weights: &[Scalar]) -> P
where
    P: Clone + Mul<Scalar, Output = P> + std::ops::AddAssign,
{
    let mut sum = slopes[0].clone() * weights[0];
    for (i, slope) in slopes.iter().enumerate().skip(1) {
        sum += slope.clone() * weights[i];
    }
    sum
}

/// Advances `point` one RKMK step from `t` to `t + dt` with the `Tab` tableau —
/// [`integrate_rkmk`] without the history, and with the stage-slope buffer
/// `scratch` passed in so a stepping loop allocates nothing per step. `scratch`
/// may start empty; its contents are overwritten.
pub fn rkmk_step<Fld, Tab, T>(
    rate: &mut impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Increment, T>, String>,
    point: &Fld::Point,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Fld::Increment>,
) -> Result<Fld::Point, IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
{
    rkmk_stage_slopes_into::<Fld, Tab, T>(rate, point, t, dt, scratch, None)?;
    reconstruct_or_err::<Fld>(point, &weight(scratch, Tab::B))
}

/// Runge–Kutta–Munthe-Kaas: a fixed-step [`ButcherTableau`] run in the field's
/// Lie algebra, with the [`IntegrableField::dexpinv`] correction per stage and a
/// single [`IntegrableField::reconstruct`] per step. Reduces to the plain tableau
/// on a flat field. See [`rkmk_step`] for the allocation-free single step.
pub fn integrate_rkmk<Fld, Tab, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Fld::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
    U: TensorVec<Item = Fld::Point>,
{
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    let mut scratch = Vec::new();
    let mut carry: Option<Derivative<Fld::Increment, T>> = None;
    points.push(point.clone());
    times.push(time[0]);
    for step in time.windows(2) {
        carry = rkmk_stage_slopes_into::<Fld, Tab, T>(
            &mut rate,
            &point,
            step[0],
            step[1] - step[0],
            &mut scratch,
            carry.as_ref(),
        )?;
        point = reconstruct_or_err::<Fld>(&point, &weight(&scratch, Tab::B))?;
        points.push(point.clone());
        times.push(step[1]);
    }
    Ok((times, points))
}

/// Adaptive RKMK: [`integrate_rkmk`] with embedded local-error control from the
/// tableau's `D` weights. `time` supplies only the span `[time[0], time[last]]`;
/// the returned times are the steps the controller accepted. The step is grown or
/// shrunk by `0.9 (tol / e)^{1/p}` (clamped to `[0.2, 5]`), and a step whose
/// error `e` exceeds `abs_tol + rel_tol ‖x_{n+1}‖` is rejected.
pub fn integrate_rkmk_adaptive<Fld, Tab, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Fld::Point) -> Result<Derivative<Fld::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Fld::Point,
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U), IntegrationError>
where
    Fld: IntegrableField,
    Tab: EmbeddedTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
    U: TensorVec<Item = Fld::Point>,
{
    if time.len() < 2 {
        return Err(IntegrationError::LengthTimeLessThanTwo);
    }
    let t_0 = time[0];
    let t_f = time[time.len() - 1];
    if t_0 >= t_f {
        return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
    }
    let exponent = 1.0 / Tab::ORDER;
    let dt_min = (t_f - t_0) * 1e-10;
    let mut t = t_0;
    let mut dt = t_f - t_0;
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    let mut slopes = Vec::new();
    let mut carry: Option<Derivative<Fld::Increment, T>> = None;
    points.push(point.clone());
    times.push(t_0);
    while t_f - t > dt_min {
        dt = dt.min(t_f - t);
        let next_carry = rkmk_stage_slopes_into::<Fld, Tab, T>(
            &mut rate,
            &point,
            t,
            dt,
            &mut slopes,
            carry.as_ref(),
        )?;
        let trial = reconstruct_or_err::<Fld>(&point, &weight(&slopes, Tab::B))?;
        let error = weight(&slopes, Tab::D).norm().value().abs();
        let tolerance = abs_tol + rel_tol * trial.norm().value();
        let accept = error <= tolerance || dt <= dt_min;
        if accept {
            t += dt;
            point = trial;
            carry = next_carry;
            points.push(point.clone());
            times.push(t);
        }
        let scale = if error > 0.0 {
            (0.9 * (tolerance / error).powf(exponent)).clamp(0.2, 5.0)
        } else {
            5.0
        };
        dt *= scale;
        if !accept && dt <= dt_min {
            return Err(IntegrationError::from(
                "the adaptive RKMK step fell below the floor".to_string(),
            ));
        }
    }
    Ok((times, points))
}

/// The `Point` type of a [`StateEvolution`] model's field.
pub type EvolvedState<M, T = Time, Y = Quantity> =
    <<M as StateEvolution<T, Y>>::Field as IntegrableField>::Point;

/// The `Increment` (Lie-algebra) type of a [`StateEvolution`] model's field.
pub type EvolvedIncrement<M, T = Time, Y = Quantity> =
    <<M as StateEvolution<T, Y>>::Field as IntegrableField>::Increment;

/// A model whose internal state evolves as a product of Lie-algebra rates,
/// ready for the field drivers. [`Self::Drive`] is the externally-imposed input
/// the rate needs beside the state (e.g. the total deformation gradient).
///
/// `Y` is only a discriminant: a model type (e.g. `Canonical`) that could carry
/// several kinds of internal state selects one impl per `Y`, so it appears here
/// even though nothing in the trait names it.
pub trait StateEvolution<T = Time, Y = Quantity>
where
    <Self::Field as IntegrableField>::Increment: Differentiate<T>,
{
    /// Geometry of the composite internal state.
    type Field: IntegrableField;
    /// The externally-imposed driving input.
    type Drive;
    /// The initial internal state.
    fn initial_state(&self) -> <Self::Field as IntegrableField>::Point;
    /// The product of Lie-algebra rates at `(time, drive, state)`.
    fn state_rate(
        &self,
        time: Quantity<T>,
        drive: &Self::Drive,
        state: &<Self::Field as IntegrableField>::Point,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, T>, String>;
}

/// Runs [`integrate_rkmk`] over a [`StateEvolution`] model, sampling `drive` at
/// each stage time and starting from the model's own initial state.
pub fn integrate_rkmk_state<M, Tab, U, T, Y>(
    model: &M,
    mut drive: impl FnMut(Quantity<T>) -> M::Drive,
    time: &[Quantity<T>],
) -> Result<(Times<T>, U), IntegrationError>
where
    M: StateEvolution<T, Y>,
    Tab: ButcherTableau,
    EvolvedState<M, T, Y>: Clone,
    EvolvedIncrement<M, T, Y>: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<EvolvedIncrement<M, T, Y>, T>:
        Mul<Quantity<T>, Output = EvolvedIncrement<M, T, Y>>,
    U: TensorVec<Item = EvolvedState<M, T, Y>>,
{
    let initial = model.initial_state();
    integrate_rkmk::<M::Field, Tab, U, T>(
        |t, state| model.state_rate(t, &drive(t), state),
        time,
        initial,
    )
}

/// Runs [`integrate_rkmk_adaptive`] over a [`StateEvolution`] model, sampling
/// `drive` at each stage time and starting from the model's own initial state.
/// `time` supplies only the span `[time[0], time[last]]`; the returned times are
/// the steps the controller accepted.
pub fn integrate_rkmk_state_adaptive<M, Tab, U, T, Y>(
    model: &M,
    mut drive: impl FnMut(Quantity<T>) -> M::Drive,
    time: &[Quantity<T>],
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U), IntegrationError>
where
    M: StateEvolution<T, Y>,
    Tab: EmbeddedTableau,
    EvolvedState<M, T, Y>: Clone,
    EvolvedIncrement<M, T, Y>: Clone + Differentiate<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<EvolvedIncrement<M, T, Y>, T>:
        Mul<Quantity<T>, Output = EvolvedIncrement<M, T, Y>>,
    U: TensorVec<Item = EvolvedState<M, T, Y>>,
{
    let initial = model.initial_state();
    integrate_rkmk_adaptive::<M::Field, Tab, U, T>(
        |t, state| model.state_rate(t, &drive(t), state),
        time,
        initial,
        abs_tol,
        rel_tol,
    )
}

/// How a Runge–Kutta integrator advances its evolving unknown from the stage
/// slopes. The blanket impl is additive — `xₙ₊₁ = xₙ + (Σ cᵢ kᵢ) Δt` — which is
/// what every flat state wants; a group-valued state overrides it to stay on its
/// manifold (RKMK: [`IntegrableField::reconstruct`] plus the [`dexpinv`] slope
/// correction).
///
/// [`dexpinv`]: IntegrableField::dexpinv
pub trait StateStep<T = Time>: Differentiate<T> + Tensor + Sized {
    /// `base` advanced by the rate combination `Σ cᵢ kᵢ` over the step `dt`.
    /// Fails only for a manifold state whose reconstruction has no solution.
    fn advance(
        base: &Self,
        rate_combination: &Derivative<Self, T>,
        dt: Quantity<T>,
    ) -> Result<Self, String>;
    /// Corrects a freshly evaluated stage rate at the rate combination `_sigma`
    /// already accumulated for that stage. The identity for a flat state.
    fn correct_stage_rate(
        _sigma: &Derivative<Self, T>,
        rate: Derivative<Self, T>,
        _dt: Quantity<T>,
    ) -> Derivative<Self, T> {
        rate
    }
    /// The embedded-error magnitude for the weighted rate combination `sum`
    /// (`Σ dᵢ kᵢ`) over the step `dt`. Flat: `‖(Σ dᵢ kᵢ) dt‖`; a manifold state
    /// measures the same increment in its algebra, without the exponential map.
    fn error_measure(sum: &Derivative<Self, T>, dt: Quantity<T>, norm: &Norm) -> Scalar;
}

impl<T, Y> StateStep<T> for Y
where
    Y: Differentiate<T> + Tensor,
    for<'a> Y: Add<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
{
    fn advance(
        base: &Self,
        rate_combination: &Derivative<Self, T>,
        dt: Quantity<T>,
    ) -> Result<Self, String> {
        Ok(rate_combination * dt + base)
    }
    fn error_measure(sum: &Derivative<Self, T>, dt: Quantity<T>, norm: &Norm) -> Scalar {
        norm.measure(&(sum * dt))
    }
}
