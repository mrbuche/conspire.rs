#[cfg(test)]
mod test;

use super::{Integrable, reconstruct_or_err};
use crate::math::{
    Derivative, Differentiable, Quantity, Scalar,
    integrate::{ButcherTableau, IntegrationError},
    optimize::{EqualityConstraint, FirstOrderRootFinding, SecondOrderOptimization},
    sparse::SparseSolver,
};
use std::ops::{AddAssign, Mul};

/// Fills `slopes` with one RKMK step's corrected stage slopes `k̃ᵢ` in the
/// field's Lie algebra: per stage combine the earlier `k̃ⱼ` by row `Aᵢ`,
/// `reconstruct` the stage point, evaluate the rate, scale by `dt`, apply
/// [`Integrable::dexpinv`] at the accumulated algebra element. `slopes` is
/// cleared first and reused, so a caller that steps in a loop allocates nothing.
/// The caller weights the entries by `B` (the step) and, for an embedded pair,
/// by `D` (the error estimate).
///
/// FSAL: `first_rate` seeds stage 0 (`C[0] == 0`) with a rate carried from the
/// previous step, skipping that evaluation; when `Tab::FSAL`, the raw rate at
/// the final stage (whose point is the step solution) is returned for the next
/// step to seed with.
pub(super) fn rkmk_stage_slopes_into<Field, Tab, T>(
    rate: &mut impl FnMut(Quantity<T>, &Field::Point) -> Result<Derivative<Field::Increment, T>, String>,
    point: &Field::Point,
    t: Quantity<T>,
    dt: Quantity<T>,
    slopes: &mut Vec<Field::Increment>,
    first_rate: Option<&Derivative<Field::Increment, T>>,
) -> Result<Option<Derivative<Field::Increment, T>>, IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
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
            Some(sigma) => reconstruct_or_err::<Field>(point, sigma)?,
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
            Some(sigma) => Field::dexpinv(sigma, increment),
            None => increment,
        });
    }
    Ok(carry)
}

pub(super) fn weight<P>(slopes: &[P], weights: &[Scalar]) -> P
where
    P: Clone + Mul<Scalar, Output = P> + AddAssign,
{
    let mut sum = slopes[0].clone() * weights[0];
    for (i, slope) in slopes.iter().enumerate().skip(1) {
        sum += slope.clone() * weights[i];
    }
    sum
}

/// Advances `point` one RKMK step from `t` to `t + dt` with the `Tab` tableau —
/// [`super::integrate_rkmk`] without the history, and with the stage-slope buffer
/// `scratch` passed in so a stepping loop allocates nothing per step. `scratch`
/// may start empty; its contents are overwritten.
pub fn rkmk_step<Field, Tab, T>(
    rate: &mut impl FnMut(Quantity<T>, &Field::Point) -> Result<Derivative<Field::Increment, T>, String>,
    point: &Field::Point,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Field::Increment>,
) -> Result<Field::Point, IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
{
    rkmk_stage_slopes_into::<Field, Tab, T>(rate, point, t, dt, scratch, None)?;
    reconstruct_or_err::<Field>(point, &weight(scratch, Tab::B))
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
pub fn rkmk_dae_step<Field, Tab, Z, T>(
    rate: &mut impl FnMut(
        Quantity<T>,
        &Field::Point,
        &Z,
    ) -> Result<Derivative<Field::Increment, T>, String>,
    solve: &mut impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<Z, String>,
    point: &Field::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Field::Increment>,
    first_rate: Option<&Derivative<Field::Increment, T>>,
) -> Result<(Field::Point, Z, Option<Derivative<Field::Increment, T>>), IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
{
    let (z_stage, carry) = rkmk_dae_stage_slopes_into::<Field, Tab, Z, T>(
        rate, solve, point, z, t, dt, scratch, first_rate,
    )?;
    let advanced = reconstruct_or_err::<Field>(point, &weight(scratch, Tab::B))?;
    let z_final = solve(t + dt, &advanced, &z_stage)?;
    Ok((advanced, z_final, carry))
}

/// [`rkmk_dae_step`] with the algebraic unknown resolved by first-order
/// root-finding at every stage abscissa, built from `function`/`jacobian`/
/// `solver` exactly as `ExplicitDaeVariableStepExplicitFirstOrderRoot` builds
/// its `solution` closure for the legacy flat DAE solver — the split between
/// root-finding and minimization is orthogonal to which field the state lives
/// on, so this is the one place that wrapping happens for the RKMK-DAE path.
/// Any [`super::StateEvolution`] model that also supplies a residual and its
/// Jacobian in terms of the *whole* field state gets the manifold-aware
/// return map for free, without hand-rolling this closure itself.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rkmk_dae_step_first_order_root<Field, Tab, F, J, Z, T>(
    rate: &mut impl FnMut(
        Quantity<T>,
        &Field::Point,
        &Z,
    ) -> Result<Derivative<Field::Increment, T>, String>,
    mut function: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<F, String>,
    mut jacobian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<J, String>,
    solver: &impl FirstOrderRootFinding<F, J, Z>,
    point: &Field::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Field::Increment>,
    first_rate: Option<&Derivative<Field::Increment, T>>,
    mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
) -> Result<(Field::Point, Z, Option<Derivative<Field::Increment, T>>), IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
{
    let mut solve = |t: Quantity<T>, point: &Field::Point, z_guess: &Z| -> Result<Z, String> {
        Ok(solver.root(
            |z| function(t, point, z),
            |z| jacobian(t, point, z),
            z_guess.clone(),
            equality_constraint(t),
            None,
        )?)
    };
    rkmk_dae_step::<Field, Tab, Z, T>(rate, &mut solve, point, z, t, dt, scratch, first_rate)
}

/// [`rkmk_dae_step`] with the algebraic unknown resolved by second-order
/// minimization at every stage abscissa, built from `function`/`jacobian`/
/// `hessian`/`solver` the same way [`rkmk_dae_step_first_order_root`] builds
/// it for root-finding — the two are siblings so a model whose equilibrium is
/// naturally posed as a potential (rather than a residual) gets the same
/// manifold-aware return map.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rkmk_dae_step_second_order_minimize<Field, Tab, F, J, H, Z, T>(
    rate: &mut impl FnMut(
        Quantity<T>,
        &Field::Point,
        &Z,
    ) -> Result<Derivative<Field::Increment, T>, String>,
    mut function: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<F, String>,
    mut jacobian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<J, String>,
    mut hessian: impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<H, String>,
    solver: &impl SecondOrderOptimization<F, J, H, Z>,
    point: &Field::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    scratch: &mut Vec<Field::Increment>,
    first_rate: Option<&Derivative<Field::Increment, T>>,
    mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    sparse: Option<SparseSolver>,
) -> Result<(Field::Point, Z, Option<Derivative<Field::Increment, T>>), IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
{
    let mut solve = |t: Quantity<T>, point: &Field::Point, z_guess: &Z| -> Result<Z, String> {
        Ok(solver.minimize(
            |z| function(t, point, z),
            |z| jacobian(t, point, z),
            |z| hessian(t, point, z),
            z_guess.clone(),
            equality_constraint(t),
            sparse.clone(),
        )?)
    };
    rkmk_dae_step::<Field, Tab, Z, T>(rate, &mut solve, point, z, t, dt, scratch, first_rate)
}

/// Fills `slopes` with one RKMK-DAE step's corrected stage slopes, resolving the
/// algebraic unknown at each stage abscissa; returns the last stage's `z` as the
/// seed for the caller's endpoint solve, and the FSAL carry (see
/// [`rkmk_dae_step`]). [`rkmk_dae_step`] without the endpoint, so an adaptive
/// driver can weight the slopes by `D` and reject a step before paying for that
/// solve.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn rkmk_dae_stage_slopes_into<Field, Tab, Z, T>(
    rate: &mut impl FnMut(
        Quantity<T>,
        &Field::Point,
        &Z,
    ) -> Result<Derivative<Field::Increment, T>, String>,
    solve: &mut impl FnMut(Quantity<T>, &Field::Point, &Z) -> Result<Z, String>,
    point: &Field::Point,
    z: &Z,
    t: Quantity<T>,
    dt: Quantity<T>,
    slopes: &mut Vec<Field::Increment>,
    first_rate: Option<&Derivative<Field::Increment, T>>,
) -> Result<(Z, Option<Derivative<Field::Increment, T>>), IntegrationError>
where
    Field: Integrable,
    Tab: ButcherTableau,
    Field::Point: Clone,
    Field::Increment: Clone + Differentiable<T>,
    Z: Clone,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
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
            Some(sigma) => reconstruct_or_err::<Field>(point, sigma)?,
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
            Some(sigma) => Field::dexpinv(sigma, increment),
            None => increment,
        });
    }
    Ok((z_stage, carry))
}
