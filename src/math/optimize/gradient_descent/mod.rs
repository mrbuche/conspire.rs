#[cfg(test)]
mod test;

use super::{
    super::{Tensor, TensorRank0},
    Dirichlet, FirstOrder, Neumann, OptimizeError,
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

impl<X: Tensor> FirstOrder<X> for GradientDescent {
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
        neumann: Option<Neumann>,
    ) -> Result<X, OptimizeError> {
        //
        // How to choose short (below, dx*dg/dg*dg) or long (dx*dx/dx*dg) steps?
        // Or even allow different options for calculating step size?
        // Like using backtracking line search with: (1), checked decrease, (2) Armijo condition (sufficient decrease), (3) trust region, etc. (see slides).
        // Those methods might also be abstracted to be used in multiple places, like if you make a nonlinear conjugate gradient solver.
        // And then within the NLCG, different formulas for beta?
        //
        let mut residual;
        let mut residual_change = initial_guess.copy() * 0.0;
        let mut solution = initial_guess;
        let mut solution_change = solution.copy();
        let mut step_size = 1e-2;
        let mut step_trial;
        if let Some(ref bc) = dirichlet {
            bc.places
                .iter()
                .zip(bc.values.iter())
                .for_each(|(place, value)| *solution.get_at_mut(place) = *value)
        }
        for _ in 0..self.max_steps {
            residual = jacobian(&solution)?;
            if let Some(ref bc) = neumann {
                bc.places
                    .iter()
                    .zip(bc.values.iter())
                    .for_each(|(place, value)| *residual.get_at_mut(place) -= value)
            }
            if let Some(ref bc) = dirichlet {
                bc.places
                    .iter()
                    .for_each(|place| *residual.get_at_mut(place) = 0.0)
            }
            if residual.norm() < self.abs_tol {
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
                residual_change = residual.copy();
                solution_change = solution.copy();
                solution -= residual * step_size;
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
}
