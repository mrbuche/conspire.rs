#[cfg(test)]
mod test;

use super::{
    super::{Hessian, Tensor, TensorRank0},
    Dirichlet, Neumann, OptimizeError, SecondOrder,
};
use crate::ABS_TOL;
use std::ops::Div;

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

impl<H: Hessian, J: Tensor, X: Tensor> SecondOrder<H, J, X> for NewtonRaphson
where
    J: Div<H, Output = X>,
{
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
        neumann: Option<Neumann>,
    ) -> Result<X, OptimizeError> {
        if let Some(bc) = dirichlet {
            // let lagrangian; // L(x,λ) = U(x) - λ(Ax - b)
            // let multipliers;
            //
            // Any appetite to make this also take nonlinear equality constraints and handle differently than linear?
            // Would you make Dirichlet an enum with linear/quadratic/nonlinear variants to match here?
            // Then can solidify the underlying BC data type maybe.
            //
            // need (X, J)::into<Vector> and H::into<Matrix>
            // or similar, also tack on multipliers to former and A to latter
            todo!()
        } else {
            //
            // might be able to pass neumann in anyway here, except for the indexing...
            // use Vec<Vec<usize>> for sparse indexing, should try to make that into a type to prevent construction issues (length mismatches, etc.)
            //
            unconstrained(self, jacobian, hessian, initial_guess, neumann)
        }
    }
}

fn unconstrained<H: Hessian, J: Tensor, X: Tensor>(
    newton: &NewtonRaphson,
    jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
    hessian: impl Fn(&X) -> Result<H, OptimizeError>,
    initial_guess: X,
    neumann: Option<Neumann>,
) -> Result<X, OptimizeError>
where
    J: Div<H, Output = X>,
{
    let mut residual;
    let mut solution = initial_guess;
    let mut tangent;
    for _ in 0..newton.max_steps {
        residual = jacobian(&solution)?;
        if let Some(ref bc) = neumann {
            bc.places
                .iter()
                .zip(bc.values.iter())
                .for_each(|(place, value)|
                    //
                    // may need a method for sparse (or at least special) indexing
                    // ndarray lets you index like A[ [i, j] ], maybe can reproduce?
                    panic!()
                    //
                    // *residual.get_at_mut(place) -= value
                )
        }
        tangent = hessian(&solution)?;
        if residual.norm() < newton.abs_tol {
            if newton.check_minimum && !tangent.is_positive_definite() {
                return Err(OptimizeError::NotMinimum(
                    format!("{}", solution),
                    format!("{:?}", &newton),
                ));
            } else {
                return Ok(solution);
            }
        } else {
            solution -= residual / tangent;
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        newton.max_steps,
        format!("{:?}", &newton),
    ))
}