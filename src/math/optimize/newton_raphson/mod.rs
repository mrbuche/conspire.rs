#[cfg(test)]
mod test;

use super::{
    super::{Hessian, Tensor, TensorRank0, SquareMatrix},
    EqualityConstraint, Dirichlet, OptimizeError, SecondOrder, SecondOrderRoot,
    LinearEqualityConstraint, IntoConstraint
};
use crate::ABS_TOL;
use std::ops::{Div, SubAssign};

/// The Newton-Raphson method.
#[derive(Debug)]
pub struct NewtonRaphson {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Whether to check if solution is minimum.
    pub check_minimum: bool,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for NewtonRaphson {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            check_minimum: true,
            max_steps: 250,
        }
    }
}

impl<C, H, J, X> SecondOrder<C, H, J, X> for NewtonRaphson
where
    C: IntoConstraint<LinearEqualityConstraint>,
    H: Hessian,
    J: Div<H, Output = X> + Tensor,
    X: Tensor,
{
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint<C>,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(constraint) => {
                let length = initial_guess.iter().count();
                let (matrix, vector) = constraint.into_constraint(length);
                let mut solution = initial_guess;
                //
                // maybe into_matrix for Hessians should take a parameters for extra space?
                // wait -> what do you need exactly for the null space method anyway?
                //
                // should be easy to automatically find Z from A such that A*Z=0, try it here!
                // maybe just leave note to figure out null space method eventually
                // and leave notes (like steps using QR to get Z)
                //
                // let lagrangian; // L(x,λ) = U(x) - λ(Ax - b)
                // let multipliers;
                todo!()
            }
            EqualityConstraint::None => {
                self.root(jacobian, hessian, initial_guess)
            }
        }
    }
}

impl<H, J, X> SecondOrderRoot<H, J, X> for NewtonRaphson
where
    H: Hessian,
    J: Div<H, Output = X> + Tensor,
    X: Tensor,
{
    fn root(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError> {
        let mut residual;
        let mut solution = initial_guess;
        let mut tangent;
        for _ in 0..self.max_steps {
            residual = jacobian(&solution)?;
            tangent = hessian(&solution)?;
            if residual.norm() < self.abs_tol {
                if self.check_minimum && !tangent.is_positive_definite() {
                    return Err(OptimizeError::NotMinimum(
                        format!("{}", solution),
                        format!("{:?}", &self),
                    ));
                } else {
                    return Ok(solution);
                }
            } else {
                solution -= residual / tangent;
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
}
