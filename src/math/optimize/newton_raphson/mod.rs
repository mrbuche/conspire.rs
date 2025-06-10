#[cfg(test)]
mod test;

use super::{
    super::{
        Banded, Hessian, Jacobian, Matrix, Scalar, Solution, SquareMatrix, Tensor, TensorVec,
        Vector,
    },
    EqualityConstraint, FirstOrderRootFinding, LineSearch, OptimizeError, Search,
    SecondOrderOptimization,
};
use crate::ABS_TOL;
use std::ops::{Div, Mul};

/// The Newton-Raphson method.
#[derive(Debug)]
pub struct NewtonRaphson {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Line search algorithm.
    pub line_search: Option<LineSearch>,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for NewtonRaphson {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            line_search: None,
            max_steps: 25,
        }
    }
}

impl<F, J, X> FirstOrderRootFinding<F, J, X> for NewtonRaphson
where
    F: Jacobian + Div<J, Output = X>,
    for<'a> &'a F: From<&'a X>,
    J: Hessian,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
    Vector: From<X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                initial_guess,
                None,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => unconstrained(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                initial_guess,
            ),
        }
    }
}

impl<J, H, X> SecondOrderOptimization<Scalar, J, H, X> for NewtonRaphson
where
    H: Hessian,
    J: Jacobian + Div<H, Output = X>,
    for<'a> &'a J: From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
    Vector: From<X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                function,
                jacobian,
                hessian,
                initial_guess,
                banded,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => {
                unconstrained(self, function, jacobian, hessian, initial_guess)
            }
        }
    }
}

fn unconstrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
    hessian: impl Fn(&X) -> Result<H, OptimizeError>,
    initial_guess: X,
) -> Result<X, OptimizeError>
where
    H: Hessian,
    J: Jacobian + Div<H, Output = X>,
    for<'a> &'a J: From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
    Vector: From<X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    let mut direction;
    let mut residual;
    let mut solution = initial_guess;
    let mut tangent;
    for _ in 0..newton_raphson.max_steps {
        residual = jacobian(&solution)?;
        tangent = hessian(&solution)?;
        if residual.norm_inf() < newton_raphson.abs_tol {
            return Ok(solution);
        } else {
            direction = residual / tangent;
            if let Some(algorithm) = &newton_raphson.line_search {
                direction *=
                    algorithm.line_search(&function, &jacobian, &solution, &direction, &1.0)?
            }
            solution -= direction
        }
    }
    Err(OptimizeError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", &newton_raphson),
    ))
}

fn constrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
    hessian: impl Fn(&X) -> Result<H, OptimizeError>,
    initial_guess: X,
    banded: Option<Banded>,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizeError>
where
    H: Hessian,
    J: Jacobian + Div<H, Output = X>,
    // for<'a> &'a J: From<&'a X>,
    X: Solution,
    // for<'a> &'a X: Mul<Scalar, Output = X>,
    Vector: From<X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    let mut direction;
    let num_variables = initial_guess.num_entries();
    let num_constraints = constraint_rhs.len();
    let num_total = num_variables + num_constraints;
    let mut multipliers = Vector::zero(num_constraints);
    let mut residual = Vector::zero(num_total);
    let mut solution = initial_guess;
    let mut tangent = SquareMatrix::zero(num_total);
    constraint_matrix
        .iter()
        .enumerate()
        .for_each(|(i, constraint_matrix_i)| {
            constraint_matrix_i
                .iter()
                .enumerate()
                .for_each(|(j, constraint_matrix_ij)| {
                    tangent[i + num_variables][j] = -constraint_matrix_ij;
                    tangent[j][i + num_variables] = -constraint_matrix_ij;
                })
        });
    for _ in 0..newton_raphson.max_steps {
        (jacobian(&solution)? - &multipliers * &constraint_matrix).fill_into_chained(
            &constraint_rhs - &constraint_matrix * &solution,
            &mut residual,
        );
        hessian(&solution)?.fill_into(&mut tangent);
        if residual.norm_inf() < newton_raphson.abs_tol {
            return Ok(solution)
        } else if let Some(ref band) = banded {
            direction = tangent.solve_lu_banded(&residual, band)?
        } else {
            direction = tangent.solve_lu(&residual)?
        }
        //
        // See now you need to have "function", "jacobian", etc. below take in {solution; multipliers} as one long vector and so on
        // I.e. "function" becomes the Lagrangian, "jacobian" the whole residual, etc.
        // Can you make another method in Search to deal with 2 arguments from constraints?
        // May be nice to continue to keep separate since separate null-space and range-space solves could be expedient.
        //
        // if let Some(algorithm) = &newton_raphson.line_search {
        //     direction *= algorithm.line_search(&function, &jacobian, &solution, &direction, &1.0)?
        // }
        solution.decrement_from_chained(&mut multipliers, direction)
        // The convexity of every step of the solves can be verified (with LDL, LL, etc.).
        // Also, consider revisiting null-space method to drastically reduce solve size.
    }
    Err(OptimizeError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", &newton_raphson),
    ))
}
