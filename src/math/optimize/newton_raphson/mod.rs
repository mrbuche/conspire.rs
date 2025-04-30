#[cfg(test)]
mod test;

use super::{
    super::{Hessian, Rank2, SquareMatrix, Tensor, TensorRank0, TensorVec, Vector},
    Dirichlet, EqualityConstraint, FirstOrderRootFinding, OptimizeError, SecondOrderOptimization,
};
use crate::ABS_TOL;
use std::ops::{Div, Sub, SubAssign};

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

// Try separate null-space and range-space steps eventually:
// Q, R = np.linalg.qr(A.T, mode='complete')
// Z = Q[:, A.T.shape[1]:]
// assert np.all(A.dot(Z) == 0)
// There are some breadcrumbs that may suggest that
// it may not work as well for large systems.
// Not sure how much truth there is to that.
// Either way, need to compare how performance scales.

// impl<C, H, J, X> SecondOrder<C, H, J, X> for NewtonRaphson
// where
//     C: ToConstraint<LinearEqualityConstraint> + Clone, // get rid of this clone!
//     H: Hessian,
//     J: Div<H, Output = X> + Tensor + Into<Vector>,
//     X: Tensor + Into<Vector> + for <'a> SubAssign<&'a [f64]>,
//     for<'a> &'a C: Sub<&'a X, Output = Vector>,
//     //
//     // finish getting Into<Vector> (both) into Tensor and take " + Into<Vector>" out
//     //
//     // also, you dont even need the "Div<H, Output = X>" feature!!!
//     //
//     // ALRIGHT THIS IS GETTING OUT OF HAND
//     // WHY PASS IN THE STRUCTURED TYPES AND TEMPLATE ALL THIS IF YOU WILL ALWAYS MAKE MATRIX AND VECTOR
//     // JUST ASSUME THAT IS DONE BEFOREHAND AND JUST WORK WITH SIMPLER THINGS IN THE CASE OF CONSTRAINTS
//     // CAN KEEP THE ORIGINAL TEMPLATING IN THE CASE OF NO CONSTRAINTS
//     // ALSO FOR (LINEAR) CONSTRAINTS, JUST PASS IN THE "A" AND "b" PERHAPS
//     //
//     // CAN IMPLEMENT SEPARATE TRAIT SequentialQuadraticProgramming FOR CONSTRAINTS CASE
//     // MAY BE ABLE TO KEEP FN DEFINITION AS JUST "minimize" SINCE ARGS WILL NOT CONFLICT
//     //
//     // AND MAYBE TRY TO GO BACK AND GET RID OF IRRELEVANT IMPLS YOU DID IN MATH
//     //
// {
//     fn minimize(
//         &self,
//         function: impl Fn(&X) -> Result<TensorRank0, OptimizeError>,
//         jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
//         hessian: impl Fn(&X) -> Result<H, OptimizeError>,
//         initial_guess: X,
//         equality_constraint: EqualityConstraint<C>,
//     ) -> Result<X, OptimizeError> {
//         match equality_constraint {
//             EqualityConstraint::Linear(constraint) => {
//                 let mut lagrangian;
//                 let (mut tangent, rhs) = constraint.to_constraint(initial_guess.iter().count());
//                 // if not need rhs do not return it, instead impl len() for EqualityConstraint and make to_constraint to_kkt or something
//                 let constraint_len = rhs.len();
//                 let solution_len = tangent.len() - constraint_len;
//                 let mut multipliers = Vector::ones(constraint_len);
//                 let mut increment;
//                 let mut residual: Vector;
//                 let mut satisfaction: Vector;
//                 let mut solution = initial_guess;
//                 for _ in 0..self.max_steps {
//                     satisfaction = &constraint - &solution;
//                     lagrangian = function(&solution)? + &multipliers * &satisfaction; // is the second part right???
//                     residual = jacobian(&solution)?.into();
//                     residual.append(&mut satisfaction);
//                     hessian(&solution)?.fill_into(&mut tangent);
//                     if residual.norm() < self.abs_tol {
//                         if self.check_minimum && !tangent.is_positive_definite() {
//                             return Err(OptimizeError::NotMinimum(
//                                 format!("{}", solution),
//                                 format!("{:?}", &self),
//                             ));
//                         } else {
//                             return Ok(solution);
//                         }
//                     } else {
//                         increment = residual / &tangent;
//                         solution -= &increment[..solution_len];
//                         multipliers -= &increment[solution_len..];
//                     }
//                 }
//                 Err(OptimizeError::MaximumStepsReached(
//                     self.max_steps,
//                     format!("{:?}", &self),
//                 ))
//             }
//             EqualityConstraint::None => {
//                 self.root(jacobian, hessian, initial_guess)
//             }
//         }
//     }
// }

impl FirstOrderRootFinding for NewtonRaphson {
    fn root<F, J, X>(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>
    where
        F: Div<J, Output = X> + Tensor,
        X: Tensor,
    {
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
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", &self),
        ))
    }
    fn solve(
        &self,
        function: impl Fn(&Vector) -> Result<Vector, OptimizeError>,
        jacobian: impl Fn(&Vector) -> Result<SquareMatrix, OptimizeError>,
        initial_guess: Vector,
        equality_constraint: EqualityConstraint,
    ) -> Result<Vector, OptimizeError> {
        match equality_constraint {
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                let num_variables = initial_guess.len();
                let num_constraints = constraint_rhs.len();
                let num_total = num_variables + num_constraints;
                let mut increment;
                let mut multipliers = Vector::ones(num_constraints);
                let mut residual = Vector::zero(num_total);
                let mut solution = initial_guess;
                // let mut state = Vector::zero(num_total);
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
                    (function(&solution)? - &multipliers * &constraint_matrix)
                        .iter()
                        .zip(residual.iter_mut())
                        .for_each(|(&function_i, residual_i)| *residual_i = function_i);
                    (&constraint_rhs - &constraint_matrix * &solution)
                        .iter()
                        .zip(residual.iter_mut().skip(num_variables))
                        .for_each(|(&satisfaction_i, residual_i)| *residual_i = satisfaction_i);
                    jacobian(&solution)?
                        .iter()
                        .zip(tangent.iter_mut())
                        .for_each(|(jacobian_i, tangent_i)| {
                            jacobian_i
                                .iter()
                                .zip(tangent_i.iter_mut())
                                .for_each(|(jacobian_ij, tangent_ij)| *tangent_ij = *jacobian_ij)
                        });

tangent.iter().for_each(|foobar|
    println!("{:?}", foobar)
);

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
                        // solution
                        //     .iter()
                        //     .zip(state.iter_mut())
                        //     .for_each(|(&solution_i, state_i)| *state_i = solution_i);
                        // multipliers
                        //     .iter()
                        //     .zip(state.iter_mut().skip(num_variables))
                        //     .for_each(|(&multipliers_i, state_i)| *state_i = multipliers_i);
                        increment = &residual / &tangent;

                        // println!("norm: {}", residual.norm());
                        // println!("residual: {:?}", residual);
                        // println!("trace: {:?}", tangent.trace());
                        // println!("inverse: {:?}", tangent.inverse());
                        println!("increment: {:?}", increment);

                        solution
                            .iter_mut()
                            .zip(increment.iter())
                            .for_each(|(solution_i, &increment_i)| *solution_i -= increment_i);
                        multipliers
                            .iter_mut()
                            .zip(increment.iter().skip(num_variables))
                            .for_each(|(multipliers_i, &increment_i)| {
                                *multipliers_i -= increment_i
                            });
                    }
                }
                Err(OptimizeError::MaximumStepsReached(
                    self.max_steps,
                    format!("{:?}", &self),
                ))
            }
            EqualityConstraint::None => self.root(function, jacobian, initial_guess),
        }
    }
}

impl<F, H, J, X> SecondOrderOptimization<F, H, J, X> for NewtonRaphson
where
    H: Hessian,
    J: Div<H, Output = X> + Tensor,
    X: Tensor,
{
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
    ) -> Result<(X, F), OptimizeError> {
        let mut potential;
        let mut residual;
        let mut solution = initial_guess;
        let mut tangent;
        for _ in 0..self.max_steps {
            potential = function(&solution)?;
            residual = jacobian(&solution)?;
            tangent = hessian(&solution)?;
            if residual.norm() < self.abs_tol {
                if self.check_minimum && !tangent.is_positive_definite() {
                    return Err(OptimizeError::NotMinimum(
                        format!("{}", solution),
                        format!("{:?}", &self),
                    ));
                } else {
                    return Ok((solution, potential));
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
    fn solve(
        &self,
        function: impl Fn(&Vector) -> Result<TensorRank0, OptimizeError>,
        jacobian: impl Fn(&Vector) -> Result<Vector, OptimizeError>,
        hessian: impl Fn(&Vector) -> Result<SquareMatrix, OptimizeError>,
        initial_guess: Vector,
        equality_constraint: EqualityConstraint,
    ) -> Result<(Vector, TensorRank0), OptimizeError> {
        todo!("try to alleviate copying from above")
        //     match equality_constraint {
        //         EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
        //             let num_variables = initial_guess.len();
        //             let num_constraints = constraint_rhs.len();
        //             let num_total = num_variables + num_constraints;
        //             let mut increment;
        //             let mut lagrangian;
        //             let mut multipliers = Vector::ones(num_constraints);
        //             let mut residual = Vector::zero(num_total);
        //             let mut satisfaction;
        //             let mut solution = initial_guess;
        //             let mut state = Vector::zero(num_total);
        //             let mut tangent = SquareMatrix::zero(num_total);
        //             constraint_matrix.iter().enumerate().for_each(|(i, constraint_matrix_i)|
        //                 constraint_matrix_i.iter().enumerate().for_each(|(j, constraint_matrix_ij)| {
        //                     tangent[i + num_variables][j] = *constraint_matrix_ij;
        //                     tangent[j][i + num_variables] = *constraint_matrix_ij;
        //                 })
        //             );
        //             for _ in 0..self.max_steps {
        //                 satisfaction = &constraint_rhs - &constraint_matrix * &solution;
        //                 lagrangian = function(&solution)? + &multipliers * &satisfaction;
        //                 jacobian(&solution)?.iter().zip(residual.iter_mut()).for_each(|(&jacobian_i, residual_i)|
        //                     *residual_i = jacobian_i
        //                 );
        //                 satisfaction.iter().zip(residual.iter_mut().skip(num_variables)).for_each(|(&satisfaction_i, residual_i)|
        //                     *residual_i = satisfaction_i
        //                 );
        //                 hessian(&solution)?.iter().zip(residual.iter_mut()).for_each(|(hessian_i, residual_i)|
        //                     hessian_i.iter().zip(residual_i.iter_mut()).for_each(|(hessian_ij, residual_ij)|
        //                         *residual_ij = *hessian_ij
        //                     )
        //                 );
        //                 if residual.norm() < self.abs_tol {
        //                     if self.check_minimum && !tangent.is_positive_definite() {
        //                         return Err(OptimizeError::NotMinimum(
        //                             format!("{}", solution),
        //                             format!("{:?}", &self),
        //                         ));
        //                     } else {
        //                         return Ok((solution, lagrangian));
        //                     }
        //                 } else {
        //                     solution.iter().zip(state.iter_mut()).for_each(|(&solution_i, state_i)|
        //                         *state_i = solution_i
        //                     );
        //                     multipliers.iter().zip(state.iter_mut().skip(num_variables)).for_each(|(&multipliers_i, state_i)|
        //                         *state_i = multipliers_i
        //                     );
        //                     increment = &state / &tangent;
        //                     solution.iter_mut().zip(increment.iter()).for_each(|(solution_i, &increment_i)|
        //                         *solution_i -= increment_i
        //                     );
        //                     multipliers.iter_mut().zip(increment.iter().skip(num_variables)).for_each(|(multipliers_i, &increment_i)|
        //                         *multipliers_i -= increment_i
        //                     );
        //                 }
        //             }
        //             Err(OptimizeError::MaximumStepsReached(
        //                 self.max_steps,
        //                 format!("{:?}", &self),
        //             ))
        //         }
        //         EqualityConstraint::None => {
        //             self.minimize(function, jacobian, hessian, initial_guess)
        //         }
        //     }
    }
}
