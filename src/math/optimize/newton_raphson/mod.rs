#[cfg(test)]
mod test;

use super::{
    super::{
        Hessian, HessianBlock, Jacobian, LuDecomposition, Matrix, Scalar, Solution, SquareMatrix,
        Tensor, Vector, sparse::SparseSolver,
    },
    BacktrackingLineSearch, EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock,
    LineSearch, OptimizationError, SecondOrderOptimization, SecondOrderOptimizationBlock,
    SolveStrategy,
};
use crate::ABS_TOL;
use crate::math::Norm;
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
    /// Norm type for error evaluation.
    pub norm: Norm,
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
            norm: Norm::Chebyshev,
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
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                initial_guess,
                sparse,
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
        function: impl FnMut(&X) -> Result<Scalar, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        match match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                function,
                jacobian,
                hessian,
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                function,
                jacobian,
                hessian,
                initial_guess,
                sparse,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => {
                unconstrained(self, function, jacobian, hessian, initial_guess)
            }
        } {
            Ok(solution) => Ok(solution),
            Err(error) => Err(OptimizationError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> FirstOrderRootFindingBlock<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>
    for NewtonRaphson
where
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a Matrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    fn root_block(
        &self,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (Matrix, Vector),
        constraint_local: (Matrix, Vector),
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError> {
        match blocked(
            self,
            |_: &U, _: &V| panic!("No line search in root finding"),
            residual_global,
            residual_local,
            tangents,
            initial_guess,
            constraint_global,
            constraint_local,
            strategy,
        ) {
            Ok(solution) => Ok(solution),
            Err(error) => Err(OptimizationError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>
    SecondOrderOptimizationBlock<Scalar, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> for NewtonRaphson
where
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a Matrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    fn minimize_block(
        &self,
        function: impl FnMut(&U, &V) -> Result<Scalar, String>,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (Matrix, Vector),
        constraint_local: (Matrix, Vector),
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError> {
        match blocked(
            self,
            function,
            residual_global,
            residual_local,
            tangents,
            initial_guess,
            constraint_global,
            constraint_local,
            strategy,
        ) {
            Ok(solution) => Ok(solution),
            Err(error) => Err(OptimizationError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

const PENALTY_SAFETY: Scalar = 2.0;

fn violation<T>(constraint_matrix: &Matrix, constraint_rhs: &Vector, variables: &T) -> Scalar
where
    for<'a> &'a Matrix: Mul<&'a T, Output = Vector>,
{
    (constraint_rhs - constraint_matrix * variables)
        .iter()
        .map(|entry| entry.abs())
        .sum()
}

fn kkt_block<K>(
    tangent: &K,
    constraint_matrix: &Matrix,
    size: usize,
    block: &mut SquareMatrix,
    offset: usize,
) where
    K: HessianBlock,
{
    tangent.fill_into_block(block, offset, offset);
    constraint_matrix
        .iter()
        .enumerate()
        .for_each(|(a, constraint_matrix_a)| {
            (0..size).for_each(|j| {
                block[offset + size + a][offset + j] = -constraint_matrix_a[j];
                block[offset + j][offset + size + a] = -constraint_matrix_a[j];
            })
        })
}

fn kkt_residual<R, T>(
    residual: R,
    multipliers: &Vector,
    constraint_matrix: &Matrix,
    constraint_rhs: &Vector,
    variables: &T,
    chained: &mut Vector,
) where
    R: Jacobian,
    for<'a> &'a Matrix: Mul<&'a T, Output = Vector>,
{
    (residual - multipliers * constraint_matrix)
        .fill_into_chained(constraint_rhs - constraint_matrix * variables, chained)
}

#[allow(clippy::too_many_arguments)]
fn blocked<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&U, &V) -> Result<Scalar, String>,
    mut residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
    mut residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
    mut tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
    initial_guess: (U, V),
    constraint_global: (Matrix, Vector),
    constraint_local: (Matrix, Vector),
    strategy: SolveStrategy,
) -> Result<(U, V), OptimizationError>
where
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a Matrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    let (mut global, mut local) = initial_guess;
    let mut penalty = 0.0 as Scalar;
    let (constraint_matrix_global, constraint_rhs_global) = constraint_global;
    let (constraint_matrix_local, constraint_rhs_local) = constraint_local;
    let num_global = global.size();
    let num_local = local.size();
    let num_outer = num_global + constraint_rhs_global.len();
    let num_inner = num_local + constraint_rhs_local.len();
    let mut multipliers_global = Vector::zero(constraint_rhs_global.len());
    let mut multipliers_local = Vector::zero(constraint_rhs_local.len());
    let eliminating = !matches!(strategy, SolveStrategy::Monolithic { elimination: false });
    let (inner, outer) = if eliminating {
        (num_inner, num_outer)
    } else {
        (0, 0)
    };
    let mut column = Vector::zero(inner);
    let mut coupling_global = Matrix::zero(outer, inner);
    let mut coupling_local = Matrix::zero(inner, outer);
    let mut eliminated = vec![Vector::zero(inner); outer.min(num_global)];
    let mut factorization = LuDecomposition::zero(inner);
    let mut monolithic = SquareMatrix::zero(if eliminating {
        0
    } else {
        num_outer + num_inner
    });
    let mut residual = Vector::zero(num_outer + num_inner);
    let mut tangent_inner = SquareMatrix::zero(inner);
    let mut tangent_outer = SquareMatrix::zero(outer);
    let mut update_inner = Vector::zero(num_inner);
    let mut update_outer = Vector::zero(num_outer);
    for _ in 0..=newton_raphson.max_steps {
        if matches!(strategy, SolveStrategy::Condensed) {
            for _ in 0..=newton_raphson.max_steps {
                kkt_residual(
                    residual_local(&global, &local)?,
                    &multipliers_local,
                    &constraint_matrix_local,
                    &constraint_rhs_local,
                    &local,
                    &mut update_inner,
                );
                if newton_raphson.norm.apply(&update_inner) < newton_raphson.abs_tol {
                    break;
                }
                let (_, _, _, tangent) = tangents(&global, &local)?;
                kkt_block(
                    &tangent,
                    &constraint_matrix_local,
                    num_local,
                    &mut tangent_inner,
                    0,
                );
                tangent_inner.factorize_lu_into(&mut factorization)?;
                let decrement = factorization.solve(&update_inner);
                local.decrement_from_chained(&mut multipliers_local, decrement)
            }
        }
        kkt_residual(
            residual_global(&global, &local)?,
            &multipliers_global,
            &constraint_matrix_global,
            &constraint_rhs_global,
            &global,
            &mut update_outer,
        );
        kkt_residual(
            residual_local(&global, &local)?,
            &multipliers_local,
            &constraint_matrix_local,
            &constraint_rhs_local,
            &local,
            &mut update_inner,
        );
        update_outer
            .iter()
            .chain(update_inner.iter())
            .zip(residual.iter_mut())
            .for_each(|(entry, residual_i)| *residual_i = *entry);
        if newton_raphson.norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok((global, local));
        }
        let (tangent_uu, tangent_vu, tangent_uv, tangent_vv) = tangents(&global, &local)?;
        let (decrement_outer, decrement_inner) = if eliminating {
            kkt_block(
                &tangent_uu,
                &constraint_matrix_global,
                num_global,
                &mut tangent_outer,
                0,
            );
            kkt_block(
                &tangent_vv,
                &constraint_matrix_local,
                num_local,
                &mut tangent_inner,
                0,
            );
            tangent_uv.fill_into_block(&mut coupling_global, 0, 0);
            tangent_vu.fill_into_block(&mut coupling_local, 0, 0);
            tangent_inner.factorize_lu_into(&mut factorization)?;
            eliminated
                .iter_mut()
                .enumerate()
                .for_each(|(k, eliminated_k)| {
                    (0..num_local).for_each(|i| column[i] = coupling_local[i][k]);
                    factorization.solve_into(&column, eliminated_k)
                });
            let offset = factorization.solve(&update_inner);
            (0..num_global).for_each(|i| {
                (0..num_local).for_each(|j| {
                    let coupling = coupling_global[i][j];
                    (0..num_global)
                        .for_each(|k| tangent_outer[i][k] -= coupling * eliminated[k][j]);
                    update_outer[i] -= coupling * offset[j]
                })
            });
            let decrement_outer = tangent_outer.solve_lu(&update_outer)?;
            let mut decrement_inner = offset;
            (0..num_global).for_each(|k| {
                decrement_inner
                    .iter_mut()
                    .zip(eliminated[k].iter())
                    .for_each(|(decrement_inner_i, eliminated_ki)| {
                        *decrement_inner_i -= eliminated_ki * decrement_outer[k]
                    })
            });
            (decrement_outer, decrement_inner)
        } else {
            kkt_block(
                &tangent_uu,
                &constraint_matrix_global,
                num_global,
                &mut monolithic,
                0,
            );
            kkt_block(
                &tangent_vv,
                &constraint_matrix_local,
                num_local,
                &mut monolithic,
                num_outer,
            );
            tangent_uv.fill_into_block(&mut monolithic, 0, num_outer);
            tangent_vu.fill_into_block(&mut monolithic, num_outer, 0);
            let decrement = monolithic.solve_lu(&residual)?;
            let mut decrement_outer = Vector::zero(num_outer);
            let mut decrement_inner = Vector::zero(num_inner);
            decrement
                .iter()
                .zip(decrement_outer.iter_mut().chain(decrement_inner.iter_mut()))
                .for_each(|(entry, decrement_i)| *decrement_i = *entry);
            (decrement_outer, decrement_inner)
        };
        let step_size = if matches!(newton_raphson.line_search, LineSearch::None) {
            1.0
        } else {
            penalty = penalty.max(
                PENALTY_SAFETY
                    * multipliers_global
                        .iter()
                        .zip(decrement_outer.iter().skip(num_global))
                        .chain(
                            multipliers_local
                                .iter()
                                .zip(decrement_inner.iter().skip(num_local)),
                        )
                        .fold(0.0, |largest: Scalar, (multiplier, decrement)| {
                            largest.max((multiplier - decrement).abs())
                        }),
            );
            let violated = penalty
                * (violation(&constraint_matrix_global, &constraint_rhs_global, &global)
                    + violation(&constraint_matrix_local, &constraint_rhs_local, &local));
            let mut gradient_global = Vector::zero(num_global);
            let mut gradient_local = Vector::zero(num_local);
            residual_global(&global, &local)?.fill_into(&mut gradient_global);
            residual_local(&global, &local)?.fill_into(&mut gradient_local);
            let slope = gradient_global
                .iter()
                .zip(decrement_outer.iter())
                .chain(gradient_local.iter().zip(decrement_inner.iter()))
                .map(|(gradient_i, decrement_i)| gradient_i * decrement_i)
                .sum::<Scalar>()
                + violated;
            let value = function(&global, &local)? + violated;
            if slope < newton_raphson.abs_tol {
                1.0
            } else {
                match newton_raphson.line_search.backtrack_merit(
                    |step| {
                        let mut trial_global = global.clone();
                        let mut trial_local = local.clone();
                        let mut trial_multipliers_global = multipliers_global.clone();
                        let mut trial_multipliers_local = multipliers_local.clone();
                        trial_global.decrement_from_chained(
                            &mut trial_multipliers_global,
                            &decrement_outer * step,
                        );
                        trial_local.decrement_from_chained(
                            &mut trial_multipliers_local,
                            &decrement_inner * step,
                        );
                        Ok(function(&trial_global, &trial_local)?
                            + penalty
                                * (violation(
                                    &constraint_matrix_global,
                                    &constraint_rhs_global,
                                    &trial_global,
                                ) + violation(
                                    &constraint_matrix_local,
                                    &constraint_rhs_local,
                                    &trial_local,
                                )))
                    },
                    value,
                    slope,
                    1.0,
                ) {
                    Ok(step_size) => step_size,
                    Err(error) => {
                        return Err(OptimizationError::Upstream(
                            format!("{error}"),
                            format!("{newton_raphson:?}"),
                        ));
                    }
                }
            }
        };
        if step_size == 1.0 {
            global.decrement_from_chained(&mut multipliers_global, decrement_outer);
            local.decrement_from_chained(&mut multipliers_local, decrement_inner)
        } else {
            global.decrement_from_chained(&mut multipliers_global, &decrement_outer * step_size);
            local.decrement_from_chained(&mut multipliers_local, &decrement_inner * step_size)
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", newton_raphson),
    ))
}

fn unconstrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
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
    for _ in 0..=newton_raphson.max_steps {
        residual = jacobian(&solution)?;
        if newton_raphson.norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else {
            tangent = hessian(&solution)?;
            decrement = &residual / tangent;
            step_size = newton_raphson.backtracking_line_search(
                &mut function,
                &mut jacobian,
                &solution,
                &residual,
                &decrement,
                1.0,
            )?;
            if step_size != 1.0 {
                decrement *= step_size
            }
            solution -= decrement
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", newton_raphson),
    ))
}

#[allow(clippy::too_many_arguments)]
fn constrained_fixed<J, H, X>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
{
    let mut retained = vec![true; initial_guess.size()];
    indices.iter().for_each(|&index| retained[index] = false);
    let unmap: Vec<usize> = retained
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index))
        .collect();
    let mut decrement;
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    for _ in 0..=newton_raphson.max_steps {
        residual = jacobian(&solution)?.retain_from(&retained);
        if newton_raphson.norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else if let Some(ref solver) = sparse {
            let hess = hessian(&solution)?;
            decrement = solver.solve(|i, j| hess.entry(unmap[i], unmap[j]), &residual)?
        } else {
            decrement = hessian(&solution)?
                .retain_from(&retained)
                .solve_lu(&residual)?
        }
        if !matches!(newton_raphson.line_search, LineSearch::None) {
            let jac = jacobian(&solution)?;
            let mut decrement_full = &solution * 0.0;
            decrement_full.decrement_from_retained(&retained, &decrement);
            decrement_full *= -1.0;
            step_size = newton_raphson.backtracking_line_search(
                &mut function,
                &mut jacobian,
                &solution,
                &jac,
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
        format!("{:?}", newton_raphson),
    ))
}

#[allow(clippy::too_many_arguments)]
fn constrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    X: Solution,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    let mut decrement;
    let mut penalty = 0.0 as Scalar;
    let num_variables = initial_guess.size();
    let num_constraints = constraint_rhs.len();
    let num_total = num_variables + num_constraints;
    let mut multipliers = Vector::zero(num_constraints);
    let mut residual = Vector::zero(num_total);
    let mut solution = initial_guess;
    let mut tangent = SquareMatrix::zero(if sparse.is_none() { num_total } else { 0 });
    if sparse.is_none() {
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
    }
    for _ in 0..=newton_raphson.max_steps {
        (jacobian(&solution)? - &multipliers * &constraint_matrix).fill_into_chained(
            &constraint_rhs - &constraint_matrix * &solution,
            &mut residual,
        );
        if newton_raphson.norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else if let Some(ref solver) = sparse {
            let hess = hessian(&solution)?;
            decrement = solver.solve(
                |i, j| {
                    if i >= num_variables {
                        -constraint_matrix[i - num_variables][j]
                    } else if j >= num_variables {
                        -constraint_matrix[j - num_variables][i]
                    } else {
                        hess.entry(i, j)
                    }
                },
                &residual,
            )?;
        } else {
            hessian(&solution)?.fill_into(&mut tangent);
            decrement = tangent.solve_lu(&residual)?
        }
        let step_size = if matches!(newton_raphson.line_search, LineSearch::None) {
            1.0
        } else {
            penalty = penalty.max(
                PENALTY_SAFETY
                    * multipliers
                        .iter()
                        .zip(decrement.iter().skip(num_variables))
                        .fold(0.0, |largest: Scalar, (multiplier, decrement_i)| {
                            largest.max((multiplier - decrement_i).abs())
                        }),
            );
            let violated = penalty * violation(&constraint_matrix, &constraint_rhs, &solution);
            let mut gradient = Vector::zero(num_variables);
            jacobian(&solution)?.fill_into(&mut gradient);
            let slope = gradient
                .iter()
                .zip(decrement.iter())
                .map(|(gradient_i, decrement_i)| gradient_i * decrement_i)
                .sum::<Scalar>()
                + violated;
            let value = function(&solution)? + violated;
            if slope < newton_raphson.abs_tol {
                1.0
            } else {
                match newton_raphson.line_search.backtrack_merit(
                    |step| {
                        let mut trial = solution.clone();
                        let mut trial_multipliers = multipliers.clone();
                        trial.decrement_from_chained(&mut trial_multipliers, &decrement * step);
                        Ok(function(&trial)?
                            + penalty * violation(&constraint_matrix, &constraint_rhs, &trial))
                    },
                    value,
                    slope,
                    1.0,
                ) {
                    Ok(step_size) => step_size,
                    Err(error) => {
                        return Err(OptimizationError::Upstream(
                            format!("{error}"),
                            format!("{newton_raphson:?}"),
                        ));
                    }
                }
            }
        };
        if step_size == 1.0 {
            solution.decrement_from_chained(&mut multipliers, decrement)
        } else {
            solution.decrement_from_chained(&mut multipliers, &decrement * step_size)
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        newton_raphson.max_steps,
        format!("{:?}", newton_raphson),
    ))
}
