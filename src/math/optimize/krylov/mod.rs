#[cfg(test)]
mod test;

use super::{OptimizationError, Precondition};
use crate::math::{
    Scalar, Style, StyledError, Tensor, Vector, assert::AssertionError, styled_error,
};

const PATIENCE: usize = 30;
const PROGRESS: Scalar = 0.9;
const ACCEPTABLE: Scalar = 1e-3;

/// Which walk through the Krylov subspace to take.
///
/// Both want the operator symmetric. They part on what else they want of it,
/// and the one that wants less of it does more work per iteration to get by.
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
}

/// An iterative linear solve, asking the operator only what it does to a
/// vector.
///
/// Nothing is factorized and nothing is assembled, so the operator is free
/// never to exist as a matrix at all. What that costs is exactness: the solve
/// stops once the residual it leaves behind is small enough against the one it
/// started from, rather than at an answer.
///
/// Symmetry is a caller-supplied guarantee either way, and neither walk can
/// tell it has been given something else.
#[derive(Clone, Copy, Debug)]
pub struct Krylov {
    /// Maximum number of iterations.
    pub max_steps: usize,
    /// Which walk to take.
    pub method: KrylovMethod,
    /// Residual tolerance, relative to the residual started from.
    pub rel_tol: Scalar,
}

impl Default for Krylov {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            method: KrylovMethod::default(),
            rel_tol: 1e-10,
        }
    }
}

impl Krylov {
    /// Solves a system given only what it does to a vector.
    ///
    /// The operator need never exist as a matrix, so this is what a system
    /// assembled from pieces — such as a domain-decomposed dual (interface)
    /// operator — is reached through.
    pub fn solve_operator(
        &self,
        apply: impl FnMut(&Vector) -> Vector,
        preconditioning: impl Precondition,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        self.walk(apply, preconditioning, right_hand_side)
    }
    fn walk(
        &self,
        apply: impl FnMut(&Vector) -> Vector,
        preconditioning: impl Precondition,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        match self.method {
            KrylovMethod::ConjugateGradients => {
                self.descend(apply, preconditioning, right_hand_side)
            }
            KrylovMethod::Minres => self.minimize_residual(apply, preconditioning, right_hand_side),
        }
    }
    /// Descends the quadratic the system is the stationary point of.
    ///
    /// Each direction is conjugate to the ones before it against the operator,
    /// so the step along it is exact and never needs revisiting, which is what
    /// lets the whole solve be a walk rather than a factorization.
    ///
    /// Positive definiteness is what that rests on, and the walk says so when
    /// it turns out to be missing: a direction of nonpositive curvature is one
    /// along which the quadratic has no minimum to step to.
    fn descend(
        &self,
        mut apply: impl FnMut(&Vector) -> Vector,
        preconditioning: impl Precondition,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        let scale = right_hand_side.norm().value();
        let mut solution = Vector::zero(right_hand_side.len());
        if scale == 0.0 {
            return Ok(solution);
        }
        let divide = |residual: &Vector| preconditioning.apply(residual);
        let mut residual = right_hand_side.clone();
        let mut preconditioned = divide(&residual);
        let mut direction = preconditioned.clone();
        let mut projection = residual.full_contraction(&preconditioned);
        let mut applied;
        let mut curvature;
        let mut next;
        let mut step;
        for _ in 0..self.max_steps {
            applied = apply(&direction);
            curvature = direction.full_contraction(&applied);
            if curvature <= 0.0 {
                return Err(KrylovError::NotPositiveDefinite(curvature));
            }
            step = projection / curvature;
            solution += &direction * step;
            residual -= applied * step;
            if residual.norm().value() <= self.rel_tol * scale {
                return Ok(solution);
            }
            preconditioned = divide(&residual);
            next = residual.full_contraction(&preconditioned);
            direction *= next / projection;
            direction += &preconditioned;
            projection = next
        }
        Err(KrylovError::MaximumStepsReached(
            self.max_steps,
            residual.norm().value() / scale,
        ))
    }
    /// Shortens the residual over the subspace reached so far.
    ///
    /// The operator is turned into a tridiagonal one over an orthonormal basis
    /// built three vectors at a time, and that small system is kept factorized
    /// by a rotation per iteration. Shortest residual is a least squares
    /// question, which has an answer whether or not the quadratic has a
    /// minimum, so nothing here asks the operator to be definite.
    ///
    /// The rotations leave the length of the residual behind as a number, but
    /// that number is the residual measured through the preconditioner rather
    /// than the residual, so it is what says when to look and not what is
    /// looked at.
    fn minimize_residual(
        &self,
        mut apply: impl FnMut(&Vector) -> Vector,
        preconditioning: impl Precondition,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        let size = right_hand_side.len();
        let mut solution = Vector::zero(size);
        let divide = |residual: &Vector| preconditioning.apply(residual);
        let mut previous = right_hand_side.clone();
        let mut current = previous.clone();
        let mut preconditioned = divide(&previous);
        let squared = previous.full_contraction(&preconditioned);
        if squared < 0.0 {
            return Err(KrylovError::PreconditionerNotPositiveDefinite(squared));
        }
        let scale = squared.sqrt();
        if scale == 0.0 {
            return Ok(solution);
        }
        let (mut cosine, mut sine) = (-1.0, 0.0);
        let mut length = scale;
        let mut off_diagonal = scale;
        let (mut previous_off, mut carried, mut trailing) = (0.0, 0.0, 0.0 as Scalar);
        let mut direction = Vector::zero(size);
        let mut older;
        let mut old = Vector::zero(size);
        let mut basis;
        let load = right_hand_side.norm().value();
        let mut watched = 1.0;
        let mut demanded = self.rel_tol;
        for step in 0..self.max_steps {
            basis = &preconditioned * off_diagonal.recip();
            preconditioned = apply(&basis);
            if step > 0 {
                preconditioned -= &previous * (off_diagonal / previous_off)
            }
            let diagonal_entry = basis.full_contraction(&preconditioned);
            preconditioned -= &current * (diagonal_entry / off_diagonal);
            previous = std::mem::replace(&mut current, preconditioned);
            preconditioned = divide(&current);
            previous_off = off_diagonal;
            let squared = current.full_contraction(&preconditioned);
            if squared < -Scalar::EPSILON * scale * scale {
                return Err(KrylovError::PreconditionerNotPositiveDefinite(squared));
            }
            off_diagonal = squared.max(0.0).sqrt();
            let reached = carried;
            let shifted = cosine * trailing + sine * diagonal_entry;
            let remaining = sine * trailing - cosine * diagonal_entry;
            carried = sine * off_diagonal;
            trailing = -cosine * off_diagonal;
            let rotated = remaining
                .hypot(off_diagonal)
                .max(Scalar::EPSILON * scale.max(1.0));
            cosine = remaining / rotated;
            sine = off_diagonal / rotated;
            older = std::mem::replace(&mut old, direction);
            direction = (basis - &older * reached - &old * shifted) * rotated.recip();
            solution += &direction * (cosine * length);
            length *= sine;
            let estimate = length.abs() / scale;
            let checkpoint = step % PATIENCE == PATIENCE - 1;
            if estimate <= demanded || checkpoint {
                let truth = (right_hand_side.clone() - apply(&solution)).norm().value() / load;
                if truth <= self.rel_tol {
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
                demanded = demanded.min(estimate * 0.1)
            }
        }
        Err(KrylovError::MaximumStepsReached(
            self.max_steps,
            (right_hand_side.clone() - apply(&solution)).norm().value() / load,
        ))
    }
}

/// Possible errors encountered during an iterative linear solve.
pub enum KrylovError {
    MaximumStepsReached(usize, Scalar),
    NotPositiveDefinite(Scalar),
    PreconditionerNotPositiveDefinite(Scalar),
    StoppedShortening(usize, Scalar),
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
                Curvature along a direction: {curvature:?}."
            ),
            Self::PreconditionerNotPositiveDefinite(squared) => format!(
                "{h}The preconditioner is not positive definite.{c}\n\
                Squared length of a residual through it: {squared:?}."
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
