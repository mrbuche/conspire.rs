#[cfg(test)]
mod test;

use super::{
    super::{
        Banded, Hessian, Jacobian, Matrix, Scalar, Solution, SquareMatrix, Tensor, TensorVec,
        Vector,
    },
    BacktrackingLineSearch, EqualityConstraint, FirstOrderRootFinding, LineSearch,
    OptimizationError, SecondOrderOptimization,
};
use crate::ABS_TOL;
use std::{
    fmt::{self, Debug, Formatter},
    ops::{Div, Mul},
};

/// The Newton-Raphson method.
pub struct NewtonRaphson {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Line search algorithm.
    pub line_search: LineSearch,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl<J, X> BacktrackingLineSearch<J, X> for NewtonRaphson {
    fn get_line_search(&self) -> &LineSearch {
        &self.line_search
    }
}

impl Debug for NewtonRaphson {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NewtonRaphson {{ abs_tol: {:?}, line_search: {}, max_steps: {:?} }}",
            self.abs_tol, self.line_search, self.max_steps
        )
    }
}

impl Default for NewtonRaphson {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            line_search: LineSearch::None,
            max_steps: 25,
        }
    }
}

impl<F, J, X> FirstOrderRootFinding<F, J, X> for NewtonRaphson
where
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X> + From<&'a X>,
    J: Hessian,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                initial_guess,
                None,
                indices,
            ),
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
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X> + From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        hessian: impl Fn(&X) -> Result<H, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                function,
                jacobian,
                hessian,
                initial_guess,
                banded,
                indices,
            ),
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
    function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
    hessian: impl Fn(&X) -> Result<H, OptimizationError>,
    initial_guess: X,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X> + From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
{
    let mut decrement;
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut tangent;
    for _ in 0..newton_raphson.max_steps {
        residual = jacobian(&solution)?;
        tangent = hessian(&solution)?;
        if residual.norm_inf() < newton_raphson.abs_tol {
            return Ok(solution);
        } else {
            decrement = &residual / tangent;
            step_size = newton_raphson.backtracking_line_search(
                &function, &jacobian, &solution, &residual, &decrement, 1.0,
            )?;
            if step_size != 1.0 {
                decrement *= step_size
            }
            solution -= decrement
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", &newton_raphson),
    ))
}

#[allow(clippy::too_many_arguments)]
fn constrained_fixed<J, H, X>(
    newton_raphson: &NewtonRaphson,
    function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
    hessian: impl Fn(&X) -> Result<H, OptimizationError>,
    initial_guess: X,
    banded: Option<Banded>,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
{
    let mut retained = vec![true; initial_guess.num_entries()];
    indices.iter().for_each(|&index| retained[index] = false);
    let mut decrement;
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut tangent;
    for _ in 0..newton_raphson.max_steps {
        residual = jacobian(&solution)?.retain_from(&retained);
        tangent = hessian(&solution)?.retain_from(&retained);
        if residual.norm_inf() < newton_raphson.abs_tol {
            return Ok(solution);
        } else if let Some(ref band) = banded {
            decrement = tangent.solve_lu_banded(&residual, band)?
        } else {
            decrement = tangent.solve_lu(&residual)?
        }
        if !matches!(newton_raphson.line_search, LineSearch::None) {
            let mut decrement_full = &solution * 0.0;
            decrement_full.decrement_from_retained(&retained, &decrement);
            decrement_full *= -1.0;
            step_size = newton_raphson.backtracking_line_search(
                &function,
                &jacobian,
                &solution,
                &jacobian(&solution)?,
                &decrement_full,
                1.0,
            )?;
            if step_size != 1.0 {
                decrement *= step_size
            }
        }
        solution.decrement_from_retained(&retained, &decrement)
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", &newton_raphson),
    ))
}

#[allow(clippy::too_many_arguments)]
fn constrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    _function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
    hessian: impl Fn(&X) -> Result<H, OptimizationError>,
    initial_guess: X,
    banded: Option<Banded>,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    X: Solution,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if !matches!(newton_raphson.line_search, LineSearch::None) {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let mut decrement;
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
            return Ok(solution);
        } else if let Some(ref band) = banded {
            decrement = tangent.solve_lu_banded(&residual, band)?
        } else {
            decrement = tangent.solve_lu(&residual)?
        }
        solution.decrement_from_chained(&mut multipliers, decrement)
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", &newton_raphson),
    ))
}
