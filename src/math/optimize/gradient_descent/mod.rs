#[cfg(test)]
mod test;

use super::{
    super::{Jacobian, Matrix, Scalar, Tensor, TensorVec, Vector},
    EqualityConstraint, FirstOrderOptimization, LineSearch, OptimizeError, ZerothOrderRootFinding,
};
use crate::ABS_TOL;
use std::ops::Mul;

/// The method of gradient descent.
#[derive(Debug)]
pub struct GradientDescent {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Lagrangian dual.
    pub dual: bool,
    /// Line search algorithm.
    pub line_search: Option<LineSearch>,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dual: false,
            line_search: None,
            max_steps: 250,
        }
    }
}

const CUTBACK_FACTOR: Scalar = 0.8;
const CUTBACK_FACTOR_MINUS_ONE: Scalar = 1.0 - CUTBACK_FACTOR;
const INITIAL_STEP_SIZE: Scalar = 1e-2;

impl<X> ZerothOrderRootFinding<X> for GradientDescent
where
    X: Jacobian,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
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
            EqualityConstraint::None => unconstrained(self, function, initial_guess, None),
        }
    }
}

impl<F, X> FirstOrderOptimization<F, X> for GradientDescent
where
    X: Jacobian,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        _function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
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
            EqualityConstraint::None => unconstrained(self, jacobian, initial_guess, None),
        }
    }
}

fn unconstrained<X>(
    gradient_descent: &GradientDescent,
    jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
    initial_guess: X,
    linear_equality_constraint: Option<(&Matrix, &Vector)>,
) -> Result<X, OptimizeError>
where
    X: Jacobian,
{
    if gradient_descent.line_search.is_some() {
        unimplemented!();
    }
    let constraint = if let Some((constraint_matrix, multipliers)) = linear_equality_constraint {
        Some(multipliers * constraint_matrix)
    } else {
        None
    };
    let mut residual;
    let mut residual_change = initial_guess.clone() * 0.0;
    let mut solution = initial_guess.clone();
    let mut solution_change = solution.clone();
    let mut step_size = INITIAL_STEP_SIZE;
    let mut step_trial;
    for _ in 0..gradient_descent.max_steps {
        residual = if let Some(ref extra) = constraint {
            jacobian(&solution)? - extra
        } else {
            jacobian(&solution)?
        };
        if residual.norm_inf() < gradient_descent.abs_tol {
            return Ok(solution);
        } else {
            solution_change -= &solution;
            residual_change -= &residual;
            step_trial =
                residual_change.full_contraction(&solution_change) / residual_change.norm_squared();
            if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                step_size = step_trial.abs()
            }
            residual_change = residual.clone();
            solution_change = solution.clone();
            solution -= residual * step_size;
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{gradient_descent:?}"),
    ))
}

fn constrained<X>(
    gradient_descent: &GradientDescent,
    jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizeError>
where
    X: Jacobian,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if gradient_descent.line_search.is_some() {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let mut residual_solution;
    let mut residual_solution_change = initial_guess.clone() * 0.0;
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
    for _ in 0..gradient_descent.max_steps {
        residual_solution = jacobian(&solution)? - &multipliers * &constraint_matrix;
        residual_multipliers = &constraint_rhs - &constraint_matrix * &solution;
        if residual_solution.norm_inf() < gradient_descent.abs_tol
            && residual_multipliers.norm_inf() < gradient_descent.abs_tol
        {
            return Ok(solution);
        } else {
            solution_change -= &solution;
            residual_solution_change -= &residual_solution;
            step_trial_solution = residual_solution_change.full_contraction(&solution_change)
                / residual_solution_change.norm_squared();
            if step_trial_solution.abs() > 0.0 && !step_trial_solution.is_nan() {
                step_size_solution = step_trial_solution.abs()
            }
            residual_solution_change = residual_solution.clone();
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
            solution -= residual_solution * step_size;
            multipliers += residual_multipliers * step_size;
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{gradient_descent:?}"),
    ))
}

fn constrained_dual<X>(
    gradient_descent: &GradientDescent,
    jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizeError>
where
    X: Jacobian,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if gradient_descent.line_search.is_some() {
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
            &jacobian,
            solution.clone(),
            Some((&constraint_matrix, &multipliers)),
        ) {
            solution = result;
            residual = &constraint_rhs - &constraint_matrix * &solution;
            if residual.norm_inf() < gradient_descent.abs_tol {
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
            multipliers -= (multipliers.clone() - &multipliers_change) * CUTBACK_FACTOR_MINUS_ONE;
            step_size *= CUTBACK_FACTOR;
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{gradient_descent:?}"),
    ))
}
