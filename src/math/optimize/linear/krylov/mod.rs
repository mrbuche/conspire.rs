#[cfg(test)]
mod test;

use crate::{
    math::{
        ContractWith, Quantity, Scalar, Style, StyledError, Tensor,
        assert::AssertionError,
        optimize::{OptimizationError, Precondition},
        styled_error,
    },
    units::{Dimensionless, UnitDiv},
};
use std::{mem::replace, ops::Mul};

const PATIENCE: usize = 30;
const PROGRESS: Quantity = Quantity::new(0.9);
const ACCEPTABLE: Quantity = Quantity::new(1e-3);
const EPSILON: Quantity = Quantity::new(Scalar::EPSILON);

/// Available Krylov subspace linear solvers.
#[derive(Clone, Copy, Debug, Default)]
pub enum KrylovMethod {
    /// Conjugate gradients, descending the quadratic the system is the
    /// stationary point of.
    ///
    /// Needs the operator positive definite, and says so when it is not. Where
    /// that holds it is the cheaper of the two, keeping three vectors rather
    /// than six.
    #[default]
    ConjugateGradients,
    /// The minimal residual method, shortening the residual over the subspace
    /// reached so far.
    ///
    /// An operator with no minimum still has a residual with a shortest
    /// length, so this asks only for symmetry and serves the systems conjugate
    /// gradients has to refuse.
    Minres,
    /// The generalized minimal residual method, restarted after the given
    /// number of iterations.
    ///
    /// Asks nothing of the operator, so it serves the nonsymmetric systems the
    /// other two have to refuse. The preconditioner acts on the right, leaving
    /// the residual it minimizes the true one, at the cost of keeping a basis
    /// of up to twice the restart length.
    Gmres(usize),
}

/// An iterative linear solver via Krylov subspaces.
#[derive(Clone, Copy, Debug)]
pub struct Krylov {
    /// Maximum number of iterations.
    pub max_steps: usize,
    /// Which Krylov method to use.
    pub method: KrylovMethod,
    /// Tolerance relative to the initial residual.
    pub rel_tol: Quantity,
}

impl Default for Krylov {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            method: KrylovMethod::default(),
            rel_tol: Quantity::new(1e-10),
        }
    }
}

impl Krylov {
    /// Iteratively solves a linear system between tensors.
    #[allow(clippy::type_complexity)]
    pub fn solve<X, F, A>(
        &self,
        apply: impl FnMut(&X) -> F,
        preconditioning: impl Precondition<F, X>,
        right_hand_side: &F,
    ) -> Result<X, KrylovError>
    where
        X: Tensor
            + ContractWith<F, Output = Quantity<A>>
            + Mul<Quantity<Dimensionless>, Output = X>,
        F: Tensor
            + ContractWith<X, Output = Quantity<A>>
            + Mul<Quantity<Dimensionless>, Output = F>,
        A: UnitDiv<A, Output = Dimensionless>,
    {
        match self.method {
            KrylovMethod::ConjugateGradients => descend(
                self.max_steps,
                self.rel_tol,
                apply,
                preconditioning,
                right_hand_side,
            ),
            KrylovMethod::Minres => minimize_residual(
                self.max_steps,
                self.rel_tol,
                apply,
                preconditioning,
                right_hand_side,
            ),
            KrylovMethod::Gmres(restart) => restarted(
                self.max_steps,
                restart,
                self.rel_tol,
                apply,
                preconditioning,
                right_hand_side,
            ),
        }
    }
}

#[allow(clippy::type_complexity)]
fn descend<X, F, A>(
    max_steps: usize,
    rel_tol: Quantity,
    mut apply: impl FnMut(&X) -> F,
    preconditioning: impl Precondition<F, X>,
    right_hand_side: &F,
) -> Result<X, KrylovError>
where
    X: Tensor + ContractWith<F, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = X>,
    F: Tensor + ContractWith<X, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = F>,
    A: UnitDiv<A, Output = Dimensionless>,
{
    let scale = right_hand_side.norm();
    let mut residual = right_hand_side.clone();
    let mut preconditioned = preconditioning.apply(&residual);
    if scale == Quantity::new(0.0) {
        return Ok(preconditioned * Quantity::new(0.0));
    }
    let mut solution = preconditioned.clone() * Quantity::new(0.0);
    let mut direction = preconditioned.clone();
    let mut projection = residual.contract_with(&preconditioned);
    for _ in 0..max_steps {
        let applied = apply(&direction);
        let curvature = direction.contract_with(&applied);
        if curvature <= Quantity::new(0.0) {
            return Err(KrylovError::NotPositiveDefinite(
                curvature.ratio(projection),
            ));
        }
        let step = projection / curvature;
        solution += direction.clone() * step;
        residual -= applied * step;
        if residual.norm().ratio(scale) <= rel_tol {
            return Ok(solution);
        }
        preconditioned = preconditioning.apply(&residual);
        let next = residual.contract_with(&preconditioned);
        direction = direction * (next / projection) + &preconditioned;
        projection = next
    }
    Err(KrylovError::MaximumStepsReached(
        max_steps,
        residual.norm().ratio(scale),
    ))
}

#[allow(clippy::type_complexity)]
fn minimize_residual<X, F, A>(
    max_steps: usize,
    rel_tol: Quantity,
    mut apply: impl FnMut(&X) -> F,
    preconditioning: impl Precondition<F, X>,
    right_hand_side: &F,
) -> Result<X, KrylovError>
where
    X: Tensor + ContractWith<F, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = X>,
    F: Tensor + ContractWith<X, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = F>,
    A: UnitDiv<A, Output = Dimensionless>,
{
    let (zero, one) = (Quantity::new(0.0), Quantity::new(1.0));
    let mut previous = right_hand_side.clone();
    let mut current = previous.clone();
    let mut preconditioned = preconditioning.apply(&previous);
    let first = previous.contract_with(&preconditioned);
    if first < Quantity::new(0.0) {
        return Err(KrylovError::PreconditionerNotPositiveDefinite(
            first.ratio(first.abs()),
        ));
    }
    let mut solution = preconditioned.clone() * zero;
    if first == Quantity::new(0.0) {
        return Ok(solution);
    }
    let (mut cosine, mut sine) = (-one, zero);
    let mut length = one;
    let mut off_diagonal = one;
    let (mut previous_off, mut carried, mut trailing) = (zero, zero, zero);
    let mut direction = solution.clone();
    let mut old = solution.clone();
    let load = right_hand_side.norm();
    let mut watched = one;
    let mut demanded = rel_tol;
    for step in 0..max_steps {
        let basis = preconditioned * (one / off_diagonal);
        let mut next = apply(&basis);
        if step > 0 {
            next -= previous.clone() * (off_diagonal / previous_off)
        }
        let diagonal_entry = basis.contract_with(&next).ratio(first);
        next -= current.clone() * (diagonal_entry / off_diagonal);
        previous = replace(&mut current, next);
        preconditioned = preconditioning.apply(&current);
        previous_off = off_diagonal;
        let squared = current.contract_with(&preconditioned).ratio(first);
        if squared < -EPSILON {
            return Err(KrylovError::PreconditionerNotPositiveDefinite(squared));
        }
        off_diagonal = squared.max(zero).sqrt();
        let reached = carried;
        let shifted = cosine * trailing + sine * diagonal_entry;
        let remaining = sine * trailing - cosine * diagonal_entry;
        carried = sine * off_diagonal;
        trailing = -cosine * off_diagonal;
        let rotated = remaining.hypot(off_diagonal).max(EPSILON);
        cosine = remaining / rotated;
        sine = off_diagonal / rotated;
        let older = replace(&mut old, direction);
        direction = (basis - older * reached - old.clone() * shifted) * (one / rotated);
        solution += direction.clone() * (cosine * length);
        length *= sine;
        let estimate = length.abs();
        let checkpoint = step % PATIENCE == PATIENCE - 1;
        if estimate <= demanded || checkpoint {
            let truth = (right_hand_side.clone() - apply(&solution))
                .norm()
                .ratio(load);
            if truth <= rel_tol {
                return Ok(solution);
            }
            if checkpoint {
                if truth > PROGRESS * watched {
                    return if truth <= ACCEPTABLE {
                        Ok(solution)
                    } else {
                        Err(KrylovError::StoppedShortening(step + 1, truth))
                    };
                }
                watched = truth
            }
            demanded = demanded.min(estimate * Quantity::<Dimensionless>::new(0.1))
        }
    }
    Err(KrylovError::MaximumStepsReached(
        max_steps,
        (right_hand_side.clone() - apply(&solution))
            .norm()
            .ratio(load),
    ))
}

#[allow(clippy::type_complexity)]
fn restarted<X, F, A>(
    max_steps: usize,
    restart: usize,
    rel_tol: Quantity,
    mut apply: impl FnMut(&X) -> F,
    preconditioning: impl Precondition<F, X>,
    right_hand_side: &F,
) -> Result<X, KrylovError>
where
    X: Tensor + ContractWith<F, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = X>,
    F: Tensor + ContractWith<X, Output = Quantity<A>> + Mul<Quantity<Dimensionless>, Output = F>,
    A: UnitDiv<A, Output = Dimensionless>,
{
    let (zero, one) = (Quantity::new(0.0), Quantity::new(1.0));
    let scale = right_hand_side.norm();
    let mut solution = preconditioning.apply(right_hand_side) * zero;
    if scale == Quantity::new(0.0) {
        return Ok(solution);
    }
    let restart = restart.max(1);
    let mut steps = 0;
    let mut relative = one;
    let mut residual = right_hand_side.clone();
    while steps < max_steps {
        let beta = residual.norm();
        let share = beta.ratio(scale);
        let mut basis = vec![residual.clone()];
        let mut weights = vec![one];
        let mut preconditioned = Vec::new();
        let mut triangle = Vec::<Vec<Quantity>>::new();
        let mut rotations = Vec::<(Quantity, Quantity)>::new();
        let mut projected = vec![one];
        let mut inner = 0;
        while inner < restart && steps < max_steps {
            preconditioned.push(preconditioning.apply(&basis[inner]));
            let mut next = apply(&preconditioned[inner]);
            let mut column = vec![zero; inner + 2];
            for (index, vector) in basis.iter().enumerate() {
                let weight = weights[index];
                let coefficient = next.contraction_over(vector, beta) / (weight * weight);
                next -= vector.clone() * coefficient;
                column[index] = coefficient * weight / weights[inner];
            }
            let weight = next.norm().ratio(beta);
            let subdiagonal = weight / weights[inner];
            column[inner + 1] = subdiagonal;
            for (row, &(cosine, sine)) in rotations.iter().enumerate() {
                let upper = cosine * column[row] + sine * column[row + 1];
                column[row + 1] = cosine * column[row + 1] - sine * column[row];
                column[row] = upper;
            }
            let length = column[inner].hypot(column[inner + 1]);
            let (cosine, sine) = if length == zero {
                (one, zero)
            } else {
                (column[inner] / length, column[inner + 1] / length)
            };
            column[inner] = length;
            column.truncate(inner + 1);
            triangle.push(column);
            rotations.push((cosine, sine));
            projected.push(-sine * projected[inner]);
            projected[inner] *= cosine;
            inner += 1;
            steps += 1;
            relative = projected[inner].abs() * share;
            if relative <= rel_tol || weight <= EPSILON {
                break;
            }
            basis.push(next);
            weights.push(weight);
        }
        let mut coefficients = vec![zero; inner];
        for row in (0..inner).rev() {
            let tail: Quantity = (row + 1..inner)
                .map(|column| triangle[column][row] * coefficients[column])
                .sum();
            coefficients[row] = (projected[row] - tail) / triangle[row][row];
        }
        for ((coefficient, weight), vector) in
            coefficients.iter().zip(&weights).zip(&preconditioned)
        {
            solution += vector.clone() * (*coefficient / *weight);
        }
        residual = right_hand_side.clone() - apply(&solution);
        relative = residual.norm().ratio(scale);
        if relative <= rel_tol {
            return Ok(solution);
        }
    }
    Err(KrylovError::MaximumStepsReached(max_steps, relative))
}

/// Possible errors encountered during an iterative linear solve.
pub enum KrylovError {
    MaximumStepsReached(usize, Quantity),
    NotPositiveDefinite(Quantity),
    PreconditionerNotPositiveDefinite(Quantity),
    StoppedShortening(usize, Quantity),
}

impl StyledError for KrylovError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::MaximumStepsReached(steps, relative) => format!(
                "{h}Maximum number of iterations ({steps}) reached.{c}\n\
                Residual relative to the one started from: {relative:?}."
            ),
            Self::NotPositiveDefinite(curvature) => format!(
                "{h}The operator is not positive definite.{c}\n\
                Curvature along a direction, relative to the residual through the preconditioner: {curvature:?}."
            ),
            Self::PreconditionerNotPositiveDefinite(squared) => format!(
                "{h}The preconditioner is not positive definite.{c}\n\
                Squared length of a residual through it, relative to the first: {squared:?}."
            ),
            Self::StoppedShortening(steps, relative) => format!(
                "{h}The residual stopped shortening after {steps} iterations.{c}\n\
                Residual relative to the one started from: {relative:?}."
            ),
        }
    }
}

styled_error!(KrylovError);

impl From<KrylovError> for OptimizationError {
    fn from(error: KrylovError) -> Self {
        Self::Upstream(error.to_string(), "Krylov".to_string())
    }
}

impl From<KrylovError> for AssertionError {
    fn from(error: KrylovError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
