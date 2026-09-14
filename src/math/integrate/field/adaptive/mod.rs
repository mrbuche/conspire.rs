#[cfg(test)]
mod test;

use super::{
    Integrable,
    hermite::{HermiteSegment, hermite_at, interpolate_hermite},
    reconstruct_or_err,
    rkmk::{rkmk_dae_stage_slopes_into, rkmk_stage_slopes_into, weight},
};
use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{ButcherTableau, EmbeddedTableau, IntegrationError, Times},
    optimize::{EqualityConstraint, FirstOrderRootFinding, SecondOrderOptimization},
    sparse::SparseSolver,
};
use std::ops::Mul;

/// Adaptive [`super::rkmk_dae_step`]: embedded local-error control from the
/// tableau's `D` weights over the span `[time[0], time[last]]`, with the same
/// controller as [`integrate_rkmk_adaptive`]. A rejected step costs no
/// endpoint constraint solve. Returns the accepted times, the state history,
/// and the matching algebraic history.
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
pub fn integrate_rkmk_dae_adaptive<Field, Tab, Z, U, V, T>(
    mut rate: impl FnMut(
        Quantity<T>,
        &Field::Point,
        &Z,
    ) -> Result<Derivative<Field::Increment, T>, String>,
    mut solve: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<Z, String>,
    time: &[Quantity<T>],
    initial_condition: (Field::Point, Z),
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U, V), IntegrationError>
where
    Field: Integrable,
    Tab: EmbeddedTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
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
    let mut carry: Option<Derivative<Field::Increment, T>> = None;
    points.push(point.clone());
    algebraics.push(z.clone());
    times.push(t_0);
    while t_f - t > dt_min {
        dt = dt.min(t_f - t);
        // A solver failure partway through a trial step (e.g. the stage or
        // accept-time algebraic solve diverging because the trial inverted an
        // element) is treated the same as an error estimate exceeding
        // tolerance: shrink dt and retry, rather than aborting the whole
        // integration. Below dt_min there is nowhere smaller left to retry
        // at, so the failure is finally propagated.
        let stage = rkmk_dae_stage_slopes_into::<Field, Tab, Z, T>(
            &mut rate,
            &mut solve,
            &point,
            &z,
            t,
            dt,
            &mut slopes,
            carry.as_ref(),
        );
        let (z_stage, next_carry) = match stage {
            Ok(stage) => stage,
            Err(error) => {
                if dt <= dt_min {
                    return Err(error);
                }
                dt *= 0.2;
                continue;
            }
        };
        let sigma = weight(&slopes, Tab::B);
        let trial = match reconstruct_or_err::<Field>(&point, &sigma) {
            Ok(trial) => trial,
            Err(error) => {
                if dt <= dt_min {
                    return Err(error);
                }
                dt *= 0.2;
                continue;
            }
        };
        let error = weight(&slopes, Tab::D).norm().value().abs();
        let tolerance = abs_tol + rel_tol * trial.norm().value();
        let accept = error <= tolerance;
        if accept {
            let t_previous = t;
            let t_next = t + dt;
            match solve(t_next, &trial, &z_stage) {
                Ok(z_next) => {
                    t = t_next;
                    z = z_next;
                    carry = next_carry;
                    if dense {
                        let slope_1 = Field::dexpinv(&sigma, &rate(t, &trial, &z)? * dt);
                        segments.push(HermiteSegment::new(
                            t_previous,
                            dt,
                            point.clone(),
                            sigma,
                            slopes[0].clone(),
                            slope_1,
                        ));
                    }
                    point = trial;
                    points.push(point.clone());
                    algebraics.push(z.clone());
                    times.push(t);
                }
                Err(error) => {
                    if dt <= dt_min {
                        return Err(IntegrationError::from(error));
                    }
                    dt *= 0.2;
                    continue;
                }
            }
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
            let point = hermite_at::<Field, T>(&segments, *time_k)?;
            guess = solve(*time_k, &point, &guess)?;
            points.push(point);
            algebraics.push(guess.clone());
        }
        Ok((Times::from(time), points, algebraics))
    } else {
        Ok((times, points, algebraics))
    }
}

/// [`integrate_rkmk_dae_adaptive`] with the algebraic unknown resolved by
/// first-order root-finding at every stage abscissa, built from
/// `function`/`jacobian`/`solver` the same way
/// [`super::rkmk_dae_step_first_order_root`] builds it for a single step.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn integrate_rkmk_dae_adaptive_first_order_root<Field, Tab, F, J, Z, U, V, T>(
    rate: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<Derivative<Field::Increment, T>, String>,
    mut function: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<F, String>,
    mut jacobian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<J, String>,
    solver: &impl FirstOrderRootFinding<F, J, Z>,
    time: &[Quantity<T>],
    initial_condition: (Field::Point, Z),
    abs_tol: Scalar,
    rel_tol: Scalar,
    mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
) -> Result<(Times<T>, U, V), IntegrationError>
where
    Field: Integrable,
    Tab: EmbeddedTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
    V: TensorVec<Item = Z>,
{
    let solve = |t: Quantity<T>, point: &Field::Point, z_guess: &Z| -> Result<Z, String> {
        Ok(solver.root(
            |z| function(t, point, z),
            |z| jacobian(t, point, z),
            z_guess.clone(),
            equality_constraint(t),
            None,
        )?)
    };
    integrate_rkmk_dae_adaptive::<Field, Tab, Z, U, V, T>(
        rate,
        solve,
        time,
        initial_condition,
        abs_tol,
        rel_tol,
    )
}

/// [`integrate_rkmk_dae_adaptive`] with the algebraic unknown resolved by
/// second-order minimization at every stage abscissa, built from
/// `function`/`jacobian`/`hessian`/`solver` the same way
/// [`super::rkmk_dae_step_second_order_minimize`] builds it for a single step.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn integrate_rkmk_dae_adaptive_second_order_minimize<Field, Tab, F, J, H, Z, U, V, T>(
    rate: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<Derivative<Field::Increment, T>, String>,
    mut function: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<F, String>,
    mut jacobian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<J, String>,
    mut hessian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<H, String>,
    solver: &impl SecondOrderOptimization<F, J, H, Z>,
    time: &[Quantity<T>],
    initial_condition: (Field::Point, Z),
    abs_tol: Scalar,
    rel_tol: Scalar,
    mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    sparse: Option<SparseSolver>,
) -> Result<(Times<T>, U, V), IntegrationError>
where
    Field: Integrable,
    Tab: EmbeddedTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
    V: TensorVec<Item = Z>,
{
    let solve = |t: Quantity<T>, point: &Field::Point, z_guess: &Z| -> Result<Z, String> {
        Ok(solver.minimize(
            |z| function(t, point, z),
            |z| jacobian(t, point, z),
            |z| hessian(t, point, z),
            z_guess.clone(),
            equality_constraint(t),
            sparse.clone(),
        )?)
    };
    integrate_rkmk_dae_adaptive::<Field, Tab, Z, U, V, T>(
        rate,
        solve,
        time,
        initial_condition,
        abs_tol,
        rel_tol,
    )
}

/// Runge–Kutta–Munthe-Kaas: a fixed-step [`ButcherTableau`] run in the field's
/// Lie algebra, with the [`Integrable::dexpinv`] correction per stage and a
/// single [`Integrable::reconstruct`] per step. Reduces to the plain tableau
/// on a flat field. See [`super::rkmk_step`] for the allocation-free single
/// step.
pub fn integrate_rkmk<Field, Tab, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Field::Point) -> Result<Derivative<Field::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Field::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
{
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    let mut scratch = Vec::new();
    let mut carry: Option<Derivative<Field::Increment, T>> = None;
    points.push(point.clone());
    times.push(time[0]);
    for step in time.windows(2) {
        carry = rkmk_stage_slopes_into::<Field, Tab, T>(
            &mut rate,
            &point,
            step[0],
            step[1] - step[0],
            &mut scratch,
            carry.as_ref(),
        )?;
        point = reconstruct_or_err::<Field>(&point, &weight(&scratch, Tab::B))?;
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
pub fn integrate_rkmk_adaptive<Field, Tab, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Field::Point) -> Result<Derivative<Field::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Field::Point,
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U), IntegrationError>
where
    Field: Integrable,
    Tab: EmbeddedTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
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
    let mut carry: Option<Derivative<Field::Increment, T>> = None;
    points.push(point.clone());
    times.push(t_0);
    while t_f - t > dt_min {
        dt = dt.min(t_f - t);
        let next_carry = rkmk_stage_slopes_into::<Field, Tab, T>(
            &mut rate,
            &point,
            t,
            dt,
            &mut slopes,
            carry.as_ref(),
        )?;
        let sigma = weight(&slopes, Tab::B);
        let trial = reconstruct_or_err::<Field>(&point, &sigma)?;
        let error = weight(&slopes, Tab::D).norm().value().abs();
        let tolerance = abs_tol + rel_tol * trial.norm().value();
        let accept = error <= tolerance;
        if accept {
            let t_previous = t;
            t += dt;
            carry = next_carry;
            if dense {
                let slope_1 = Field::dexpinv(&sigma, &rate(t, &trial)? * dt);
                segments.push(HermiteSegment::new(
                    t_previous,
                    dt,
                    point.clone(),
                    sigma,
                    slopes[0].clone(),
                    slope_1,
                ));
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
            interpolate_hermite::<Field, U, T>(&segments, time)?,
        ))
    } else {
        Ok((times, points))
    }
}
