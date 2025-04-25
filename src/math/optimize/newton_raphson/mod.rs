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
        //
        // might want X to impl into::<Vector> and H into::<Matrix> for this to work instead?
        // but that sort of hurts the templating, and simple cases like scalars
        //
        // can make it call the actual method directly or not based on if there are constraints
        // with constraints: flatten J & H into Vector & Matrix, each with effects of constraints built in to new function calls, then call the method
        // and then base method called by either can just be regular unconstrained NM like below!
        //
        let x_tot_len = 5; // need x total (flattened) length method
        let foo = if let Some(bc) = dirichlet {
            let mut foo = vec![vec![0.0; bc.places.len()]; x_tot_len];
            bc.places.iter().enumerate().zip(bc.values.iter()).for_each(|((index, &place), &value)| foo[index][place] = value);
            Some(foo)
        } else {
            None
        };
        //
        // let lagrangian; // L(x,λ) = U(x) - λ(Ax - b)
        // let multipliers;
        //
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
