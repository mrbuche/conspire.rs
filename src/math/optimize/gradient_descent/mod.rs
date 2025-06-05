#[cfg(test)]
mod test;

use super::{
    super::{Jacobian, Matrix, Tensor, TensorRank0, TensorVec, Vector},
    EqualityConstraint, FirstOrderOptimization, OptimizeError, ZerothOrderRootFinding,
};
use crate::ABS_TOL;
use std::ops::Mul;

/// The method of gradient descent.
#[derive(Debug)]
pub struct GradientDescent {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            max_steps: 250,
        }
    }
}

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
                let num_constraints = constraint_rhs.len();
                let mut multipliers = Vector::ones(num_constraints) * self.abs_tol;
                let mut multipliers_change = multipliers.clone();
                let mut residual;
                let mut residual_change = Vector::zero(num_constraints);
                let mut solution = initial_guess;
                let mut step_size = 1e-2;
                let mut step_trial;
                for _ in 0..self.max_steps {
                    if let Ok(result) = gradient_descent(
                        self,
                        &function,
                        &solution,
                        Some((&constraint_matrix, &multipliers)),
                    ) {
                        solution = result;
                        residual = &constraint_rhs - &constraint_matrix * &solution;
                        // println!(
                        //     "Residual norm: {:?}, step size: {:?}",
                        //     residual.norm_inf(),
                        //     step_size
                        // );
                        if residual.norm_inf() < self.abs_tol {
                            return Ok(solution);
                        } else {
                            multipliers_change -= &multipliers;
                            residual_change -= &residual;
                            step_trial = residual_change.full_contraction(&multipliers_change)
                                / residual_change.norm_squared();
                            if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                                step_size = step_trial.abs()
                                // } else {
                                //     step_size *= 1.1
                            }
                            residual_change = residual.clone();
                            multipliers_change = multipliers.clone();
                            multipliers += residual * step_size;
                        }
                    } else {
                        // println!("CUT BACK");
                        multipliers -= (multipliers.clone() - &multipliers_change) * 0.8;
                        step_size *= 0.8;
                    }
                }
                Err(OptimizeError::MaximumStepsReached(
                    self.max_steps,
                    format!("{:?}", self),
                ))
            }
            EqualityConstraint::None => gradient_descent(self, function, &initial_guess, None),
        }
    }
}

impl<F, X> FirstOrderOptimization<F, X> for GradientDescent
where
    X: Jacobian,
{
    fn minimize(
        &self,
        _function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(_constraint_matrix, _constraint_rhs) => {
                unimplemented!("This may work with gradient ascent on multipliers, or not at all.")
            }
            EqualityConstraint::None => gradient_descent(self, jacobian, &initial_guess, None),
        }
    }
}

fn gradient_descent<X: Jacobian>(
    gradient_descent: &GradientDescent,
    jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
    initial_guess: &X,
    linear_equality_constraint: Option<(&Matrix, &Vector)>,
) -> Result<X, OptimizeError> {
    let constraint = if let Some((constraint_matrix, multipliers)) = linear_equality_constraint {
        Some(multipliers * constraint_matrix)
    } else {
        None
    };
    let mut residual;
    let mut residual_change = initial_guess.clone() * 0.0;
    let mut solution = initial_guess.clone();
    let mut solution_change = solution.clone();
    let mut step_size = 1e-2;
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
                // } else {
                //     step_size *= 1.1
            }
            residual_change = residual.clone();
            solution_change = solution.clone();
            solution -= residual * step_size;
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{:?}", gradient_descent),
    ))
}
