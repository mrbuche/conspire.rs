#[cfg(test)]
mod test;

use super::{
    super::{
        Banded, Hessian, Jacobian, Matrix, Solution, SquareMatrix, Tensor, TensorRank0, TensorVec,
        Vector,
    },
    EqualityConstraint, FirstOrderRootFinding, OptimizeError, SecondOrderOptimization,
};
use crate::ABS_TOL;
use std::ops::{Div, Mul};

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

impl<F, J, X> FirstOrderRootFinding<F, J, X> for NewtonRaphson
where
    F: Jacobian + Div<J, Output = X>,
    J: Hessian,
    X: Solution,
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
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                let num_variables = initial_guess.num_entries();
                let num_constraints = constraint_rhs.len();
                let num_total = num_variables + num_constraints;
                let mut multipliers = Vector::ones(num_constraints) * self.abs_tol;
                let mut residual = Vector::zero(num_total);
                let mut solution = initial_guess;
                let mut tangent = SquareMatrix::zero(num_total);
                for _ in 0..self.max_steps {
                    (function(&solution)? - &multipliers * &constraint_matrix).fill_into_chained(
                        &constraint_rhs - &constraint_matrix * &solution,
                        &mut residual,
                    );
                    jacobian(&solution)?.fill_into(&mut tangent);
                    constraint_matrix
                        .iter()
                        .enumerate()
                        .for_each(|(i, constraint_matrix_i)| {
                            constraint_matrix_i.iter().enumerate().for_each(
                                |(j, constraint_matrix_ij)| {
                                    tangent[i + num_variables][j] = -constraint_matrix_ij;
                                    tangent[j][i + num_variables] = -constraint_matrix_ij;
                                },
                            )
                        });
                    tangent
                        .iter_mut()
                        .skip(num_variables)
                        .for_each(|tangent_i| {
                            tangent_i
                                .iter_mut()
                                .skip(num_variables)
                                .for_each(|tangent_ij| *tangent_ij = 0.0)
                        });
                    if residual.norm() < self.abs_tol {
                        return Ok(solution);
                    } else {
                        // println!("{:?}", residual.norm());
                        // solution
                        //     .decrement_from_chained(&mut multipliers, tangent.solve_ldl(&residual)?)
                        solution
                            .decrement_from_chained(&mut multipliers, tangent.solve_lu(&residual)?)
                    }
                }
            }
            EqualityConstraint::None => {
                let mut residual;
                let mut solution = initial_guess;
                let mut tangent;
                for _ in 0..self.max_steps {
                    residual = function(&solution)?;
                    tangent = jacobian(&solution)?;
                    if residual.norm() < self.abs_tol {
                        return Ok(solution);
                    } else {
                        solution -= residual / tangent;
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

impl<F, H, J, X> SecondOrderOptimization<F, H, J, X> for NewtonRaphson
where
    H: Hessian,
    J: Jacobian + Div<H, Output = X>,
    X: Solution,
    Vector: From<X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        _function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<X, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                let num_variables = initial_guess.num_entries();
                let num_constraints = constraint_rhs.len();
                let num_total = num_variables + num_constraints;
                let mut multipliers = Vector::ones(num_constraints) * self.abs_tol;
                let mut residual = Vector::zero(num_total);
                let mut solution = initial_guess;
                let mut tangent = SquareMatrix::zero(num_total);
                constraint_matrix
                    .iter()
                    .enumerate()
                    .for_each(|(i, constraint_matrix_i)| {
                        constraint_matrix_i.iter().enumerate().for_each(
                            |(j, constraint_matrix_ij)| {
                                tangent[i + num_variables][j] = -constraint_matrix_ij;
                                tangent[j][i + num_variables] = -constraint_matrix_ij;
                            },
                        )
                    });
                for _ in 0..self.max_steps {
                    (jacobian(&solution)? - &multipliers * &constraint_matrix).fill_into_chained(
                        &constraint_rhs - &constraint_matrix * &solution,
                        &mut residual,
                    );
                    hessian(&solution)?.fill_into(&mut tangent);
                    if residual.norm() < self.abs_tol {
                        //
                        // Just wait for LDL* version of Cholesky to do verification by looking at the inertia.
                        // And should you check every single time? Seems that SQP suggests each step is verifiable.
                        // And note, the entries of D are NOT the eigenvalues.
                        //
                        // if tangent.verify(null_space) {
                        return Ok(solution);
                        // } else {
                        //     return Err(OptimizeError::NotMinimum(
                        //         format!("{}", solution),
                        //         format!("{:?}", &self),
                        //     ));
                        // }
                    } else if let Some(ref band) = banded {
                        solution.decrement_from_chained(
                            &mut multipliers,
                            tangent.solve_lu_banded(&residual, band)?,
                        )
                    } else {
                        solution
                            .decrement_from_chained(&mut multipliers, tangent.solve_lu(&residual)?)
                    }
                }
            }
            EqualityConstraint::None => {
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
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
}
