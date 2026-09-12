#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorError, TensorRank2, TensorTuple,
    TensorVec,
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

/// One RKMK step of a semi-explicit DAE: the differential field advances on its
/// manifold while the algebraic unknown is re-solved from its constraint at
/// every stage abscissa.
///
/// [`rkmk_step`] freezes the drive across the whole window, which caps the
/// coupling at first order however the two legs are ordered. Here `solve`
/// supplies `z` at each stage time `t + cᵢ Δt` from the stage point, so the
/// drive is resolved *within* the window — the half-explicit RK treatment of an
/// index-1 DAE, but with the state leg kept on its group.
///
/// `solve` is seeded with the previous stage's `z` and must return a `z`
/// satisfying the constraint at the stage it is given; the returned `z` is the
/// one consistent with the step's own endpoint.
///
/// FSAL: `first_rate` seeds stage 0 with a rate carried from the previous
/// step, skipping both that rate evaluation and its constraint solve (`z`
/// stays the `z` passed in, which is already consistent with `(t, point)`);
/// when `Tab::FSAL`, the raw rate at the final stage is returned for the next
/// step to seed with.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rkmk_dae_step<Fld, Tab, Z, T>(
    rate: &mut impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Derivative<Fld::Increment, T>, String>,
    solve: &mut impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Z, String>,
    point: &Fld::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Fld::Increment>,
    first_rate: Option<&Derivative<Fld::Increment, T>>,
) -> Result<(Fld::Point, Z, Option<Derivative<Fld::Increment, T>>), IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
{
    let (z_stage, carry) = rkmk_dae_stage_slopes_into::<Fld, Tab, Z, T>(
        rate, solve, point, z, t, dt, scratch, first_rate,
    )?;
    let advanced = reconstruct_or_err::<Fld>(point, &weight(scratch, Tab::B))?;
    let z_final = solve(t + dt, &advanced, &z_stage)?;
    Ok((advanced, z_final, carry))
}

/// Fills `slopes` with one RKMK-DAE step's corrected stage slopes, resolving the
/// algebraic unknown at each stage abscissa; returns the last stage's `z` as the
/// seed for the caller's endpoint solve, and the FSAL carry (see
/// [`rkmk_dae_step`]). [`rkmk_dae_step`] without the endpoint, so an adaptive
/// driver can weight the slopes by `D` and reject a step before paying for that
/// solve.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn rkmk_dae_stage_slopes_into<Fld, Tab, Z, T>(
    rate: &mut impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Derivative<Fld::Increment, T>, String>,
    solve: &mut impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Z, String>,
    point: &Fld::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    slopes: &mut Vec<Fld::Increment>,
    first_rate: Option<&Derivative<Fld::Increment, T>>,
) -> Result<(Z, Option<Derivative<Fld::Increment, T>>), IntegrationError>
where
    Fld: IntegrableField,
    Tab: ButcherTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
{
    slopes.clear();
    slopes.reserve(Tab::STAGES);
    let mut z_stage = z.clone();
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
        let t_stage = t + dt * Tab::C[i];
        let increment = match (i, first_rate) {
            (0, Some(seed)) => seed * dt,
            _ => {
                z_stage = solve(t_stage, &stage_point, &z_stage)?;
                let raw = rate(t_stage, &stage_point, &z_stage)?;
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
    Ok((z_stage, carry))
}

/// Cubic Hermite dense output over one accepted step, built in the field's Lie
/// algebra rather than on the state itself.
///
/// The flat interpolant combines `y_{n}`, `y_{n+1}` and the two end rates
/// affinely. That is meaningless for a group-valued state: the two states are
/// different group elements and the two rates live in different tangent spaces,
/// so the combination leaves the manifold (`det F_p` drifts) exactly the way the
/// additive march did.
///
/// Anchor everything at the left endpoint instead. The step already produced
/// `sigma` with `reconstruct(base, sigma) = y_{n+1}`, and the algebra is a flat
/// vector space — so the Hermite polynomial is built *there*,
///
/// ```math
/// \sigma(\theta) = h_{10}(\theta)\,\dot\sigma_0 + h_{01}(\theta)\,\sigma
///                + h_{11}(\theta)\,\dot\sigma_1
/// ,\qquad
/// \mathbf{y}(\theta) = \mathrm{reconstruct}(\mathbf{y}_n, \sigma(\theta))
/// ```
///
/// with `σ(0) = 0` and `σ(1) = sigma`, so both endpoints are reproduced exactly
/// and every interior point is an `expm` of a trace-free element — on the group
/// by construction. `slope_0` is the raw stage-0 slope (`dexpinv` at zero
/// displacement is the identity); `slope_1` is the endpoint rate pulled back
/// through [`IntegrableField::dexpinv`] at `sigma`, both already scaled by the
/// step. On a [`Flat`] field `reconstruct` adds and `dexpinv` is the identity,
/// and `h_{00} + h_{01} = 1` collapses this to the usual flat formula.
pub struct HermiteSegment<Fld: IntegrableField, T = Time> {
    t_0: Quantity<T>,
    h: Quantity<T>,
    base: Fld::Point,
    sigma: Fld::Increment,
    slope_0: Fld::Increment,
    slope_1: Fld::Increment,
}

impl<Fld, T> HermiteSegment<Fld, T>
where
    Fld: IntegrableField,
{
    /// A segment of the accepted step `[t_0, t_0 + h]` from `base`, the algebra
    /// displacement `sigma` over it, and the step-scaled algebra rates at its
    /// two ends (`slope_1` already pulled back through
    /// [`IntegrableField::dexpinv`] at `sigma`).
    pub fn new(
        t_0: Quantity<T>,
        h: Quantity<T>,
        base: Fld::Point,
        sigma: Fld::Increment,
        slope_0: Fld::Increment,
        slope_1: Fld::Increment,
    ) -> Self {
        Self {
            t_0,
            h,
            base,
            sigma,
            slope_0,
            slope_1,
        }
    }
    /// The state at `time`, reconstructed from the algebra Hermite polynomial.
    pub fn evaluate(&self, time: Quantity<T>) -> Result<Fld::Point, IntegrationError> {
        let theta = (time - self.t_0).value() / self.h.value();
        let theta_2 = theta * theta;
        let theta_3 = theta_2 * theta;
        let mut increment = self.slope_0.clone() * (theta_3 - 2.0 * theta_2 + theta);
        increment += self.sigma.clone() * (3.0 * theta_2 - 2.0 * theta_3);
        increment += self.slope_1.clone() * (theta_3 - theta_2);
        reconstruct_or_err::<Fld>(&self.base, &increment)
    }
}

/// Evaluates `segments` at `time_k`, in the segment that contains it (the last
/// one for a time past the final accepted step).
fn hermite_at<Fld, T>(
    segments: &[HermiteSegment<Fld, T>],
    time_k: Quantity<T>,
) -> Result<Fld::Point, IntegrationError>
where
    Fld: IntegrableField,
{
    segments
        .iter()
        .find(|segment| time_k <= segment.t_0 + segment.h)
        .unwrap_or(&segments[segments.len() - 1])
        .evaluate(time_k)
}

/// [`hermite_at`] over a whole grid of requested times.
pub fn interpolate_hermite<Fld, U, T>(
    segments: &[HermiteSegment<Fld, T>],
    time: &[Quantity<T>],
) -> Result<U, IntegrationError>
where
    Fld: IntegrableField,
    U: TensorVec<Item = Fld::Point>,
{
    let mut points = U::new();
    for time_k in time {
        points.push(hermite_at::<Fld, T>(segments, *time_k)?);
    }
    Ok(points)
}

/// Adaptive [`rkmk_dae_step`]: embedded local-error control from the tableau's
/// `D` weights over the span `[time[0], time[last]]`, with the same controller as
/// [`integrate_rkmk_adaptive`]. A rejected step costs no endpoint constraint
/// solve. Returns the accepted times, the state history, and the matching
/// algebraic history.
///
/// Dense output follows the convention of the flat DAE loop: `time` of length
/// two supplies only the span and the accepted steps are reported, while a
/// longer `time` is a list of requested report times. Each of those is served by
/// the geodesic [`HermiteSegment`] of the accepted step containing it, so the
/// reported state is on the manifold at every requested time and not only at the
/// accepted ones; the algebraic unknown is then re-solved from its constraint
/// there, warm-started along the grid. Building the segments costs one extra
/// rate evaluation per accepted step, so it is skipped when not requested.
#[allow(clippy::type_complexity)]
pub fn integrate_rkmk_dae_adaptive<Fld, Tab, Z, U, V, T>(
    mut rate: impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Derivative<Fld::Increment, T>, String>,
    mut solve: impl FnMut(Quantity<T>, &Fld::Point, &Z) -> Result<Z, String>,
    time: &[Quantity<T>],
    initial_condition: (Fld::Point, Z),
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U, V), IntegrationError>
where
    Fld: IntegrableField,
    Tab: EmbeddedTableau,
    Fld::Point: Clone,
    Fld::Increment: Clone + Differentiate<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Fld::Increment, T>: Mul<Quantity<T>, Output = Fld::Increment>,
    U: TensorVec<Item = Fld::Point>,
    V: TensorVec<Item = Z>,
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
    let dense = time.len() > 2;
    let (mut point, mut z) = initial_condition;
    let z_0 = z.clone();
    let mut points = U::new();
    let mut algebraics = V::new();
    let mut times = Times::new();
    let mut slopes = Vec::new();
    let mut segments = Vec::new();
    let mut carry: Option<Derivative<Fld::Increment, T>> = None;
    points.push(point.clone());
    algebraics.push(z.clone());
    times.push(t_0);
    while t_f - t > dt_min {
        dt = dt.min(t_f - t);
        let (z_stage, next_carry) = rkmk_dae_stage_slopes_into::<Fld, Tab, Z, T>(
            &mut rate,
            &mut solve,
            &point,
            &z,
            t,
            dt,
            &mut slopes,
            carry.as_ref(),
        )?;
        let sigma = weight(&slopes, Tab::B);
        let trial = reconstruct_or_err::<Fld>(&point, &sigma)?;
        let error = weight(&slopes, Tab::D).norm().value().abs();
        let tolerance = abs_tol + rel_tol * trial.norm().value();
        let accept = error <= tolerance || dt <= dt_min;
        if accept {
            let t_previous = t;
            t += dt;
            z = solve(t, &trial, &z_stage)?;
            carry = next_carry;
            if dense {
                let slope_1 = Fld::dexpinv(&sigma, &rate(t, &trial, &z)? * dt);
                segments.push(HermiteSegment::<Fld, T> {
                    t_0: t_previous,
                    h: dt,
                    base: point.clone(),
                    sigma,
                    slope_0: slopes[0].clone(),
                    slope_1,
                });
            }
            point = trial;
            points.push(point.clone());
            algebraics.push(z.clone());
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
                "the adaptive RKMK-DAE step fell below the floor".to_string(),
            ));
        }
    }
    if dense {
        let mut points = U::new();
        let mut algebraics = V::new();
        let mut guess = z_0;
        for time_k in time {
            let point = hermite_at::<Fld, T>(&segments, *time_k)?;
            guess = solve(*time_k, &point, &guess)?;
            points.push(point);
            algebraics.push(guess.clone());
        }
        Ok((Times::from(time), points, algebraics))
    } else {
        Ok((times, points, algebraics))
    }
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
/// tableau's `D` weights. The step is grown or shrunk by `0.9 (tol / e)^{1/p}`
/// (clamped to `[0.2, 5]`), and a step whose error `e` exceeds
/// `abs_tol + rel_tol ‖x_{n+1}‖` is rejected.
///
/// Dense output follows the convention of [`integrate_rkmk_dae_adaptive`]: `time`
/// of length two supplies only the span and the accepted steps are reported,
/// while a longer `time` is a list of requested report times, each served by the
/// geodesic [`HermiteSegment`] of the accepted step containing it. Building the
/// segments costs one extra rate evaluation per accepted step, so it is skipped
/// when not requested.
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
    let dense = time.len() > 2;
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    let mut slopes = Vec::new();
    let mut segments = Vec::new();
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
        let sigma = weight(&slopes, Tab::B);
        let trial = reconstruct_or_err::<Fld>(&point, &sigma)?;
        let error = weight(&slopes, Tab::D).norm().value().abs();
        let tolerance = abs_tol + rel_tol * trial.norm().value();
        let accept = error <= tolerance || dt <= dt_min;
        if accept {
            let t_previous = t;
            t += dt;
            carry = next_carry;
            if dense {
                let slope_1 = Fld::dexpinv(&sigma, &rate(t, &trial)? * dt);
                segments.push(HermiteSegment::<Fld, T> {
                    t_0: t_previous,
                    h: dt,
                    base: point.clone(),
                    sigma,
                    slope_0: slopes[0].clone(),
                    slope_1,
                });
            }
            point = trial;
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
    if dense {
        Ok((
            Times::from(time),
            interpolate_hermite::<Fld, U, T>(&segments, time)?,
        ))
    } else {
        Ok((times, points))
    }
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

//
// `StateStep` used to sit here: a seam meant to let a group-valued state
// override the additive Runge–Kutta march. It could never work. Its slope was
// typed `Derivative<Self, T>`, and `Differentiate` admits exactly one
// `Derivative` per state — for `(F_p, Y)` that is the group velocity `Ḟ_p`
// (`Intermediate ← Reference`), while RKMK needs the algebra element `D_p`
// (`Intermediate ← Intermediate`) for the same state. No impl can supply a
// second slope type, so the manifold branch the trait advertised was
// unreachable (the coherence error it surfaced as was only a symptom).
//
// Manifold stepping instead dispatches on the field — `IntegrableField`, whose
// `Point`/`Increment` split carries exactly that distinction and which no state
// type can collide with. The additive march is now inline in the two
// Runge–Kutta loops that used the trait.
//
