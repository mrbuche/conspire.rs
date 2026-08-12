#[cfg(test)]
mod test;

use super::{
    super::{Jacobian, Matrix, Scalar, Solution, Tensor, Vector},
    BacktrackingLineSearch, EqualityConstraint, FirstOrderOptimization, LineSearch,
    OptimizationError, StepSize, ZerothOrderRootFinding,
};
use crate::ABS_TOL;
use crate::math::unit::{UnitDiv, UnitMul, UnitSum};
use crate::math::{Erase, Is, Norm};
use std::{
    fmt::{self, Debug, Formatter},
    ops::Mul,
};

const CUTBACK_FACTOR: Scalar = 0.8;
const CUTBACK_FACTOR_MINUS_ONE: Scalar = 1.0 - CUTBACK_FACTOR;
const INITIAL_STEP_SIZE: Scalar = 1e-2;

/// The method of gradient descent.
pub struct GradientDescent {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Lagrangian dual.
    pub dual: bool,
    /// Norm type for error evaluation.
    pub error_norm: Norm,
    /// Line search algorithm.
    pub line_search: LineSearch,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance.
    pub rel_tol: Option<Scalar>,
}

impl<J, X> BacktrackingLineSearch<J, X> for GradientDescent {
    fn get_line_search(&self) -> &LineSearch {
        &self.line_search
    }
}

impl Debug for GradientDescent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientDescent {{ abs_tol: {:?}, dual: {:?}, line_search: {}, max_steps: {:?}, rel_tol: {:?} }}",
            self.abs_tol, self.dual, self.line_search, self.max_steps, self.rel_tol
        )
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dual: false,
            error_norm: Norm::Chebyshev,
            line_search: LineSearch::None,
            max_steps: 250,
            rel_tol: None,
        }
    }
}

impl<F, X, E> ZerothOrderRootFinding<F, X> for GradientDescent
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding."),
                function,
                initial_guess,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                if self.dual {
                    constrained_dual(
                        self,
                        function,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                } else {
                    constrained(
                        self,
                        function,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                }
            }
            EqualityConstraint::None => unconstrained(
                self,
                |_: &X| panic!("No line search in root finding."),
                function,
                initial_guess,
                None,
            ),
        }
    }
}

impl<F, J, X, E> FirstOrderOptimization<F, J, X> for GradientDescent
where
    F: Erase<Erased = Scalar> + Tensor,
    <J as Tensor>::Unit: UnitMul<<X as Tensor>::Unit>,
    <<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output: UnitSum,
    <<<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    J: Erase<Erased = E> + Jacobian + Mul<StepSize<J, X>, Output = X>,
    for<'a> &'a J: Mul<StepSize<J, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<J as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        mut function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError> {
        // The line search compares merits against a constraint violation, which
        // is a number, so the objective spends its unit here rather than at
        // every call site that has one to hand over.
        let objective = move |argument: &X| function(argument).map(|value| *value.erase());
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                constrained_fixed(self, objective, jacobian, initial_guess, indices)
            }
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                if self.dual {
                    constrained_dual(
                        self,
                        jacobian,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                } else {
                    constrained(
                        self,
                        jacobian,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                }
            }
            EqualityConstraint::None => {
                unconstrained(self, objective, jacobian, initial_guess, None)
            }
        }
    }
}

fn unconstrained<F, X, E>(
    gradient_descent: &GradientDescent,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    linear_equality_constraint: Option<(&Matrix, &Vector)>,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
{
    let constraint = if let Some((constraint_matrix, multipliers)) = linear_equality_constraint {
        Some(multipliers * constraint_matrix)
    } else {
        None
    };
    let mut residual;
    let mut residual_change = None;
    let mut solution = initial_guess.clone();
    let mut solution_change = solution.clone();
    let mut step_size = INITIAL_STEP_SIZE;
    let mut step_trial;
    let mut steps = 0;
    loop {
        residual = if let Some(ref extra) = constraint {
            jacobian(&solution)? - extra
        } else {
            jacobian(&solution)?
        };
        if gradient_descent.error_norm.measure(&residual) < gradient_descent.abs_tol {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            solution_change -= &solution;
            let change = residual_change.get_or_insert_with(|| zeroed(&residual));
            *change -= &residual;
            // The step size carries the unit of the solution over that of the
            // residual, which is what makes the increment below dimensionally
            // sound without either of them having to be unitless.
            step_trial =
                change.erase().full_contraction(solution_change.erase()) / change.norm_squared();
            if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                step_size = step_trial.abs()
            }
            step_size = gradient_descent.backtracking_line_search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &residual,
                step_size,
            )?;
            *change = residual.clone();
            solution_change = solution.clone();
            solution -= residual * StepSize::<F, X>::new(step_size);
        }
    }
}

fn constrained_fixed<F, X, E>(
    gradient_descent: &GradientDescent,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
{
    let mut relative_scale = 0.0;
    let mut residual: F;
    let mut residual_change = None;
    let mut residual_norm;
    let mut solution = initial_guess.clone();
    let mut solution_change = solution.clone();
    let mut step_size = INITIAL_STEP_SIZE;
    let mut step_trial;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        residual.zero_out(&indices);
        residual_norm = gradient_descent.error_norm.measure(&residual);
        if gradient_descent.rel_tol.is_some() && steps == 0 {
            relative_scale = gradient_descent.error_norm.measure(&residual)
        }
        if residual_norm < gradient_descent.abs_tol {
            return Ok(solution);
        } else if let Some(rel_tol) = gradient_descent.rel_tol
            && residual_norm / relative_scale < rel_tol
        {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            solution_change -= &solution;
            let change = residual_change.get_or_insert_with(|| zeroed(&residual));
            *change -= &residual;
            // The step size carries the unit of the solution over that of the
            // residual, which is what makes the increment below dimensionally
            // sound without either of them having to be unitless.
            step_trial =
                change.erase().full_contraction(solution_change.erase()) / change.norm_squared();
            if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                step_size = step_trial.abs()
            }
            step_size = gradient_descent.backtracking_line_search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &residual,
                step_size,
            )?;
            *change = residual.clone();
            solution_change = solution.clone();
            solution -= residual * StepSize::<F, X>::new(step_size);
        }
    }
}

fn constrained<F, X, E>(
    gradient_descent: &GradientDescent,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if !matches!(gradient_descent.line_search, LineSearch::None) {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let mut residual_solution;
    let mut residual_solution_change = None;
    let mut solution = initial_guess.clone();
    let mut solution_change = solution.clone();
    let mut step_size_solution = INITIAL_STEP_SIZE;
    let mut step_trial_solution;
    let num_constraints = constraint_rhs.len();
    let mut residual_multipliers;
    let mut residual_multipliers_change = Vector::zero(num_constraints);
    let mut multipliers = Vector::zero(num_constraints);
    let mut multipliers_change = Vector::zero(num_constraints);
    let mut step_size_multipliers = INITIAL_STEP_SIZE;
    let mut step_trial_multipliers;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual_solution = jacobian(&solution)? - &multipliers * &constraint_matrix;
        residual_multipliers = &constraint_rhs - &constraint_matrix * &solution;
        if gradient_descent.error_norm.measure(&residual_solution) < gradient_descent.abs_tol
            && gradient_descent.error_norm.measure(&residual_multipliers) < gradient_descent.abs_tol
        {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            solution_change -= &solution;
            let change = residual_solution_change.get_or_insert_with(|| zeroed(&residual_solution));
            *change -= &residual_solution;
            step_trial_solution =
                change.erase().full_contraction(solution_change.erase()) / change.norm_squared();
            if step_trial_solution.abs() > 0.0 && !step_trial_solution.is_nan() {
                step_size_solution = step_trial_solution.abs()
            }
            *change = residual_solution.clone();
            solution_change = solution.clone();
            multipliers_change -= &multipliers;
            residual_multipliers_change -= &residual_multipliers;
            step_trial_multipliers = residual_multipliers_change
                .full_contraction(&multipliers_change)
                / residual_multipliers_change.norm_squared();
            if step_trial_multipliers.abs() > 0.0 && !step_trial_multipliers.is_nan() {
                step_size_multipliers = step_trial_multipliers.abs()
            }
            residual_multipliers_change = residual_multipliers.clone();
            multipliers_change = multipliers.clone();
            step_size = step_size_solution.min(step_size_multipliers);
            solution -= residual_solution * StepSize::<F, X>::new(step_size);
            multipliers += residual_multipliers * step_size;
        }
    }
}

fn constrained_dual<F, X, E>(
    gradient_descent: &GradientDescent,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if !matches!(gradient_descent.line_search, LineSearch::None) {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let num_constraints = constraint_rhs.len();
    let mut multipliers = Vector::zero(num_constraints);
    let mut multipliers_change = multipliers.clone();
    let mut residual;
    let mut residual_change = Vector::zero(num_constraints);
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut step_trial;
    for _ in 0..gradient_descent.max_steps {
        if let Ok(result) = unconstrained(
            gradient_descent,
            |_: &X| {
                panic!("Line search needs the exact penalty function in constrained optimization.")
            },
            &mut jacobian,
            solution.clone(),
            Some((&constraint_matrix, &multipliers)),
        ) {
            solution = result;
            residual = &constraint_rhs - &constraint_matrix * &solution;
            if gradient_descent.error_norm.measure(&residual) < gradient_descent.abs_tol {
                return Ok(solution);
            } else {
                multipliers_change -= &multipliers;
                residual_change -= &residual;
                step_trial = residual_change.full_contraction(&multipliers_change)
                    / residual_change.norm_squared();
                if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                    step_size = step_trial.abs()
                }
                residual_change = residual.clone();
                multipliers_change = multipliers.clone();
                multipliers += residual * step_size;
            }
        } else {
            //
            // This sort of acts like LineSearch::Error, does it not?
            //
            multipliers -= (multipliers.clone() - &multipliers_change) * CUTBACK_FACTOR_MINUS_ONE;
            step_size *= CUTBACK_FACTOR;
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{gradient_descent:?}"),
    ))
}

/// A zeroed copy, to start the difference the step size is estimated from.
fn zeroed<F>(residual: &F) -> F
where
    F: Tensor,
{
    let mut zero = residual.clone();
    zero *= 0.0;
    zero
}
