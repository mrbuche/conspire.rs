use super::super::{
    super::{
        Erase, Hessian, HessianBlock, Is, Jacobian, LuDecomposition, Matrix, Scalar, Solution,
        Tensor, Vector,
        sparse::{CscMatrix, SparseSolver},
    },
    Direct, EqualityConstraint, EqualityConstraints, LinearSolver, ObjectiveMerit, Optimization,
    OptimizationBlock, OptimizationError, OptimizationIncremental, ResidualMerit, RootFinding,
    RootFindingBlock, RootFindingIncremental, Rows, SolveStrategy, StepSize, no_objective,
    no_objective_blocks,
};
use crate::units::{UnitMul, UnitRatio, UnitSum};
use std::{
    fmt::Display,
    ops::{Div, Mul},
};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, Objective, Residually};
use super::blocked::blocked;
use super::constrained::constrained;
use super::constrained_fixed::constrained_fixed;
use super::unconstrained::unconstrained;
use super::verify::{unchecked, verified};

impl<S, F, J, X, E> RootFinding<F, J, X> for NewtonRaphson<Direct, S>
where
    S: ResidualMerit,
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    F: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        iterate(
            self,
            Residually(&self.line_search),
            no_objective,
            function,
            jacobian,
            no_update,
            initial_guess,
            equality_constraint,
            sparse,
            false,
            false,
        )
    }
}

impl<L, S, F, J, X, E> RootFinding<F, J, X> for NewtonRaphson<L, S>
where
    S: ResidualMerit,
    L: LinearSolver<Tangent = J>,
    F: Jacobian,
    F: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X>,
{
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        iterate_linear(
            self,
            Residually(&self.line_search),
            no_objective,
            function,
            jacobian,
            no_update,
            initial_guess,
            equality_constraint,
        )
    }
}

impl<S, F, J, X, E> RootFindingIncremental<F, J, X> for NewtonRaphson<Direct, S>
where
    S: ResidualMerit,
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    F: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn root_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        iterate(
            self,
            Residually(&self.line_search),
            no_objective,
            function,
            jacobian,
            update,
            initial_guess,
            equality_constraint,
            sparse,
            false,
            true,
        )
    }
}

impl<S, F, J, H, X, E> Optimization<F, J, H, X> for NewtonRaphson<Direct, S>
where
    S: ObjectiveMerit,
    F: Erase<Erased = Scalar> + Tensor,
    <J as Tensor>::Unit: UnitMul<<X as Tensor>::Unit>,
    <<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output: UnitSum,
    <<<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        mut function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let function = move |argument: &X| function(argument).map(|value| *value.erase());
        iterate(
            self,
            Objective(&self.line_search),
            function,
            jacobian,
            hessian,
            no_update,
            initial_guess,
            equality_constraint,
            sparse,
            true,
            false,
        )
    }
}

impl<L, S, F, J, H, X, E> Optimization<F, J, H, X> for NewtonRaphson<L, S>
where
    S: ObjectiveMerit,
    L: LinearSolver<Tangent = H>,
    F: Erase<Erased = Scalar> + Tensor,
    <J as Tensor>::Unit: UnitMul<<X as Tensor>::Unit>,
    <<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output: UnitSum,
    <<<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    J: Jacobian,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X>,
{
    fn minimize(
        &self,
        mut function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let function = move |argument: &X| function(argument).map(|value| *value.erase());
        iterate_linear(
            self,
            Objective(&self.line_search),
            function,
            jacobian,
            hessian,
            no_update,
            initial_guess,
            equality_constraint,
        )
    }
}

impl<S, F, J, H, X, E> OptimizationIncremental<F, J, H, X> for NewtonRaphson<Direct, S>
where
    S: ObjectiveMerit,
    F: Erase<Erased = Scalar> + Tensor,
    <J as Tensor>::Unit: UnitMul<<X as Tensor>::Unit>,
    <<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output: UnitSum,
    <<<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn minimize_incremental(
        &self,
        mut function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let function = move |argument: &X| function(argument).map(|value| *value.erase());
        iterate(
            self,
            Objective(&self.line_search),
            function,
            jacobian,
            hessian,
            update,
            initial_guess,
            equality_constraint,
            sparse,
            true,
            true,
        )
    }
}

impl<S, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> RootFindingBlock<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>
    for NewtonRaphson<Direct, S>
where
    S: ResidualMerit,
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    fn root_block(
        &self,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (CscMatrix, Vector),
        constraint_local: (CscMatrix, Vector),
        sparse: Option<SparseSolver>,
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError> {
        blocked(
            self,
            Residually(&self.line_search),
            no_objective_blocks,
            false,
            residual_global,
            residual_local,
            tangents,
            initial_guess,
            constraint_global,
            constraint_local,
            sparse,
            strategy,
        )
        .map_err(|error| OptimizationError::upstream(error, self))
    }
}

impl<S, F, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> OptimizationBlock<F, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>
    for NewtonRaphson<Direct, S>
where
    S: ObjectiveMerit,
    F: Erase<Erased = Scalar> + Tensor,
    <Ru as Tensor>::Unit: UnitMul<<U as Tensor>::Unit>,
    <<Ru as Tensor>::Unit as UnitMul<<U as Tensor>::Unit>>::Output: UnitSum,
    <<<Ru as Tensor>::Unit as UnitMul<<U as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    <Rv as Tensor>::Unit: UnitMul<<V as Tensor>::Unit>,
    <<Rv as Tensor>::Unit as UnitMul<<V as Tensor>::Unit>>::Output: UnitSum,
    <<<Rv as Tensor>::Unit as UnitMul<<V as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    fn minimize_block(
        &self,
        mut function: impl FnMut(&U, &V) -> Result<F, String>,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (CscMatrix, Vector),
        constraint_local: (CscMatrix, Vector),
        sparse: Option<SparseSolver>,
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError> {
        let function = move |global: &U, local: &V| function(global, local).map(|v| *v.erase());
        blocked(
            self,
            Objective(&self.line_search),
            function,
            true,
            residual_global,
            residual_local,
            tangents,
            initial_guess,
            constraint_global,
            constraint_local,
            sparse,
            strategy,
        )
        .map_err(|error| OptimizationError::upstream(error, self))
    }
}

pub(super) fn no_update<X>(_: &X, _: &Vector, _: Scalar, _: bool) -> Result<(), String> {
    Ok(())
}

/// Iterates with the direct solver, in whichever form the equality constraints
/// call for.
///
/// The residual is what is driven to zero and the tangent is what it is
/// stepped by, which are the jacobian and hessian of a minimization and the
/// function and jacobian of a root. An unconstrained solution has no chained
/// vector to lend the increment through, so is only taken when it is not lent.
#[expect(clippy::too_many_arguments)]
pub(super) fn iterate<B, S, J, H, X, E>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    search: B,
    objective: impl FnMut(&X) -> Result<Scalar, String>,
    jacobian: impl FnMut(&X) -> Result<J, String>,
    hessian: impl FnMut(&X) -> Result<H, String>,
    update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    mut initial_guess: X,
    equality_constraint: EqualityConstraints,
    sparse: Option<SparseSolver>,
    minimizing: bool,
    lent: bool,
) -> Result<X, OptimizationError>
where
    B: Backtrack,
    S: Display,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    match EqualityConstraint::split(equality_constraint, &mut initial_guess) {
        (Some(indices), Rows::None) => constrained_fixed(
            newton_raphson,
            search,
            objective,
            jacobian,
            hessian,
            update,
            initial_guess,
            direct(sparse.clone()),
            minimizing,
            verified(sparse),
            indices,
        ),
        (indices, Rows::Dense(constraint_matrix, constraint_rhs)) => constrained(
            newton_raphson,
            search,
            objective,
            jacobian,
            hessian,
            update,
            initial_guess,
            minimizing,
            sparse,
            &indices.unwrap_or_default(),
            constraint_matrix,
            constraint_rhs,
        ),
        (indices, Rows::Sparse(constraint_matrix, constraint_rhs)) => constrained(
            newton_raphson,
            search,
            objective,
            jacobian,
            hessian,
            update,
            initial_guess,
            minimizing,
            sparse,
            &indices.unwrap_or_default(),
            constraint_matrix,
            constraint_rhs,
        ),
        (None, Rows::None) => {
            if lent {
                unimplemented!(
                    "An unconstrained solution has no chained vector to lend the increment through."
                )
            }
            unconstrained(
                newton_raphson,
                search,
                objective,
                jacobian,
                hessian,
                initial_guess,
                sparse,
                minimizing,
            )
        }
    }
    .map_err(|error| OptimizationError::upstream(error, newton_raphson))
}

/// Iterates with a linear solver of the user's, which needs a fixed equality
/// constraint.
#[expect(clippy::too_many_arguments)]
pub(super) fn iterate_linear<B, L, S, J, H, X, E>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: B,
    objective: impl FnMut(&X) -> Result<Scalar, String>,
    jacobian: impl FnMut(&X) -> Result<J, String>,
    hessian: impl FnMut(&X) -> Result<H, String>,
    update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    mut initial_guess: X,
    equality_constraint: EqualityConstraints,
) -> Result<X, OptimizationError>
where
    B: Backtrack,
    L: LinearSolver<Tangent = H>,
    S: Display,
    J: Jacobian,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X>,
{
    match EqualityConstraint::split(equality_constraint, &mut initial_guess) {
        (Some(indices), Rows::None) => constrained_fixed(
            newton_raphson,
            search,
            objective,
            jacobian,
            hessian,
            update,
            initial_guess,
            |tangent, _, retained, residual, decrement| {
                *decrement = newton_raphson
                    .linear_solver
                    .solve(tangent, retained, residual)?;
                Ok(())
            },
            false,
            unchecked,
            indices,
        ),
        _ => Err(OptimizationError::Intermediate(
            "This linear solver requires a fixed equality constraint.".to_string(),
        )),
    }
    .map_err(|error| OptimizationError::upstream(error, newton_raphson))
}

pub(super) fn direct<H>(
    sparse: Option<SparseSolver>,
) -> impl FnMut(H, &[bool], &[usize], &Vector, &mut Vector) -> Result<(), OptimizationError>
where
    H: Hessian,
{
    let mut factorization = None;
    move |hessian, retained, unmap, residual, decrement| {
        if let Some(ref solver) = sparse {
            *decrement = solver.solve(|i, j| hessian.entry(unmap[i], unmap[j]), residual)?
        } else {
            let factorization =
                factorization.get_or_insert_with(|| LuDecomposition::zero(unmap.len()));
            hessian
                .retain_from(retained)
                .factorize_lu_into(factorization)?;
            factorization.solve_into(residual, decrement)
        }
        Ok(())
    }
}
