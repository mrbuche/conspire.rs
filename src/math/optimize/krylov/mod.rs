#[cfg(test)]
mod test;

use super::{OptimizationError, Precondition};
use crate::math::{
    Scalar, Style, StyledError, Tensor, Vector, assert::AssertionError, styled_error,
};
use std::mem::replace;

const PATIENCE: usize = 30;
const PROGRESS: Scalar = 0.9;
const ACCEPTABLE: Scalar = 1e-3;

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
    /// Iteratively solves a linear system given its action on a vector.
    pub fn solve(
        &self,
        apply: impl FnMut(&Vector) -> Vector,
        preconditioning: impl Precondition,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
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

fn descend(
    max_steps: usize,
    rel_tol: Scalar,
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
    for _ in 0..max_steps {
        applied = apply(&direction);
        curvature = direction.full_contraction(&applied);
        if curvature <= 0.0 {
            return Err(KrylovError::NotPositiveDefinite(curvature));
        }
        step = projection / curvature;
        solution += &direction * step;
        residual -= applied * step;
        if residual.norm().value() <= rel_tol * scale {
            return Ok(solution);
        }
        preconditioned = divide(&residual);
        next = residual.full_contraction(&preconditioned);
        direction *= next / projection;
        direction += &preconditioned;
        projection = next
    }
    Err(KrylovError::MaximumStepsReached(
        max_steps,
        residual.norm().value() / scale,
    ))
}

fn minimize_residual(
    max_steps: usize,
    rel_tol: Scalar,
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
    let mut demanded = rel_tol;
    for step in 0..max_steps {
        basis = &preconditioned * off_diagonal.recip();
        preconditioned = apply(&basis);
        if step > 0 {
            preconditioned -= &previous * (off_diagonal / previous_off)
        }
        let diagonal_entry = basis.full_contraction(&preconditioned);
        preconditioned -= &current * (diagonal_entry / off_diagonal);
        previous = replace(&mut current, preconditioned);
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
        older = replace(&mut old, direction);
        direction = (basis - &older * reached - &old * shifted) * rotated.recip();
        solution += &direction * (cosine * length);
        length *= sine;
        let estimate = length.abs() / scale;
        let checkpoint = step % PATIENCE == PATIENCE - 1;
        if estimate <= demanded || checkpoint {
            let truth = (right_hand_side.clone() - apply(&solution)).norm().value() / load;
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
            demanded = demanded.min(estimate * 0.1)
        }
    }
    Err(KrylovError::MaximumStepsReached(
        max_steps,
        (right_hand_side.clone() - apply(&solution)).norm().value() / load,
    ))
}

fn restarted(
    max_steps: usize,
    restart: usize,
    rel_tol: Scalar,
    mut apply: impl FnMut(&Vector) -> Vector,
    preconditioning: impl Precondition,
    right_hand_side: &Vector,
) -> Result<Vector, KrylovError> {
    let scale = right_hand_side.norm().value();
    let mut solution = Vector::zero(right_hand_side.len());
    if scale == 0.0 {
        return Ok(solution);
    }
    let restart = restart.max(1);
    let mut steps = 0;
    let mut relative = 1.0;
    let mut residual = right_hand_side.clone();
    while steps < max_steps {
        let beta = residual.norm().value();
        let mut basis = vec![&residual * beta.recip()];
        let mut preconditioned = Vec::new();
        let mut triangle = Vec::<Vec<Scalar>>::new();
        let mut rotations = Vec::<(Scalar, Scalar)>::new();
        let mut projected = vec![beta];
        let mut inner = 0;
        while inner < restart && steps < max_steps {
            preconditioned.push(preconditioning.apply(&basis[inner]));
            let mut next = apply(&preconditioned[inner]);
            let mut column = vec![0.0; inner + 2];
            for (entry, vector) in column.iter_mut().zip(&basis) {
                *entry = next.full_contraction(vector);
                next -= vector * *entry;
            }
            let subdiagonal = next.norm().value();
            column[inner + 1] = subdiagonal;
            for (row, &(cosine, sine)) in rotations.iter().enumerate() {
                let upper = cosine * column[row] + sine * column[row + 1];
                column[row + 1] = cosine * column[row + 1] - sine * column[row];
                column[row] = upper;
            }
            let length = column[inner].hypot(column[inner + 1]);
            let (cosine, sine) = if length == 0.0 {
                (1.0, 0.0)
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
            relative = projected[inner].abs() / scale;
            if relative <= rel_tol || subdiagonal <= Scalar::EPSILON * beta {
                break;
            }
            basis.push(next * subdiagonal.recip());
        }
        let mut coefficients = vec![0.0; inner];
        for row in (0..inner).rev() {
            let tail: Scalar = (row + 1..inner)
                .map(|column| triangle[column][row] * coefficients[column])
                .sum();
            coefficients[row] = (projected[row] - tail) / triangle[row][row];
        }
        for (coefficient, vector) in coefficients.iter().zip(&preconditioned) {
            solution += vector * *coefficient;
        }
        residual = right_hand_side.clone() - apply(&solution);
        relative = residual.norm().value() / scale;
        if relative <= rel_tol {
            return Ok(solution);
        }
    }
    Err(KrylovError::MaximumStepsReached(max_steps, relative))
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
