#[cfg(test)]
mod test;

use super::{
    super::{Tensor, TensorRank0},
    EqualityConstraint, FirstOrderOptimization, OptimizeError, ZerothOrderRootFinding,
};
use crate::ABS_TOL;

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
    X: Tensor,
{
    fn root(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(_constraint_matrix, _constraint_rhs) => {
                unimplemented!("This may work with gradient ascent on multipliers, or not at all.")
            }
            EqualityConstraint::None => {
                let mut residual;
                let mut residual_change = initial_guess.clone() * 0.0;
                let mut solution = initial_guess;
                let mut solution_change = solution.clone();
                let mut step_size = 1e-2;
                let mut step_trial;
                for _ in 0..self.max_steps {
                    residual = function(&solution)?;
                    if residual.norm_inf() < self.abs_tol {
                        return Ok(solution);
                    } else {
                        solution_change -= &solution;
                        residual_change -= &residual;
                        step_trial = residual_change.full_contraction(&solution_change)
                            / residual_change.norm_squared();
                        if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                            step_size = step_trial.abs()
                        } else {
                            step_size *= 1.1
                        }
                        residual_change = residual.clone();
                        solution_change = solution.clone();
                        solution -= residual * step_size;
                    }
                }
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
}

impl<F, X> FirstOrderOptimization<F, X> for GradientDescent
where
    X: Tensor,
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
            EqualityConstraint::None => {
                //
                // How to choose short (below, dx*dg/dg*dg) or long (dx*dx/dx*dg) steps?
                // Or even allow different options for calculating step size?
                // Like using backtracking line search with: (1), checked decrease, (2) Armijo condition (sufficient decrease), (3) Wolfe, (4) trust region, etc. (see slides).
                // Those methods might also be abstracted to be used in multiple places, like if you make a nonlinear conjugate gradient solver.
                // And then within the NLCG, different formulas for beta?
                //
                let mut residual;
                let mut residual_change = initial_guess.clone() * 0.0;
                let mut solution = initial_guess;
                let mut solution_change = solution.clone();
                let mut step_size = 1e-2;
                let mut step_trial;
                for _ in 0..self.max_steps {
                    residual = jacobian(&solution)?;
                    if residual.norm_inf() < self.abs_tol {
                        return Ok(solution);
                    } else {
                        solution_change -= &solution;
                        residual_change -= &residual;
                        step_trial = residual_change.full_contraction(&solution_change)
                            / residual_change.norm_squared();
                        if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                            step_size = step_trial.abs()
                        } else {
                            step_size *= 1.1
                        }
                        residual_change = residual.clone();
                        solution_change = solution.clone();
                        solution -= residual * step_size;
                    }
                }
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
}
