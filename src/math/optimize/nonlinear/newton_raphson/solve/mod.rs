use crate::math::{
    ContractWith, Erase, Hessian, HessianBlock, Is, Jacobian, LuDecomposition, Matrix, Quantity,
    Scalar, Solution, Tensor, Vector,
    optimize::{
        Direct, EqualityConstraint, EqualityConstraints, LinearSolver, ObjectiveMerit,
        Optimization, OptimizationBlock, OptimizationError, OptimizationIncremental, ResidualMerit,
        RootFinding, RootFindingBlock, RootFindingIncremental, Rows, SolveStrategy, StepSize,
        no_objective, no_objective_blocks,
    },
    sparse::{CscMatrix, SparseSolver},
};
use crate::units::{Dimensionless, UnitMul, UnitRatio, UnitSum};
use std::{
    fmt::Display,
    ops::{Div, Mul},
};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, Gradient, Objective, Residually};
use super::blocked::blocked;
use super::constrained::constrained;
use super::fixed::constrained_fixed;
use super::unconstrained::unconstrained;
use super::verify::verified;

impl<S, F, J, X> RootFinding<F, J, X> for NewtonRaphson<Direct, S>
where
    S: ResidualMerit,
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    X: Jacobian + Solution,
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
            no_objective::<X, Scalar>,
            function,
            jacobian,
            no_update,
            initial_guess,
            equality_constraint,
            sparse,
            false,
        )
    }
}

impl<L, S, F, J, X> RootFinding<F, J, X> for NewtonRaphson<L, S>
where
    S: ResidualMerit,
    L: LinearSolver<Tangent = J, Residual = F, Decrement = X>,
    F: Jacobian,
    X: Jacobian + Solution,
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
            no_objective::<X, Scalar>,
            function,
            jacobian,
            no_update,
            initial_guess,
            equality_constraint,
        )
    }
}

impl<S, F, J, X> RootFindingIncremental<F, J, X> for NewtonRaphson<Direct, S>
where
    S: ResidualMerit,
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    X: Jacobian + Solution,
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
            no_objective::<X, Scalar>,
            function,
            jacobian,
            update,
            initial_guess,
            equality_constraint,
            sparse,
            true,
        )
    }
}

impl<S, F, J, H, X, U> Optimization<F, J, H, X> for NewtonRaphson<Direct, S>
where
    S: ObjectiveMerit,
    J: ContractWith<X, Output = Quantity<U>>,
    Dimensionless: UnitMul<U, Output = U>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X>,
    F: Erase<Erased = Scalar> + Into<Quantity<U>> + Tensor,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    X: Jacobian + Solution,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
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
            false,
        )
    }
}

impl<L, S, F, J, H, X, U> Optimization<F, J, H, X> for NewtonRaphson<L, S>
where
    S: ObjectiveMerit,
    J: ContractWith<X, Output = Quantity<U>>,
    Dimensionless: UnitMul<U, Output = U>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X>,
    L: LinearSolver<Tangent = H, Residual = J, Decrement = X>,
    F: Erase<Erased = Scalar> + Into<Quantity<U>> + Tensor,
    J: Jacobian,
    X: Jacobian + Solution,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X>,
{
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
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

impl<S, F, J, H, X, U> OptimizationIncremental<F, J, H, X> for NewtonRaphson<Direct, S>
where
    S: ObjectiveMerit,
    J: ContractWith<X, Output = Quantity<U>>,
    Dimensionless: UnitMul<U, Output = U>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X>,
    F: Erase<Erased = Scalar> + Into<Quantity<U>> + Tensor,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    X: Jacobian + Solution,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
    for<'a> &'a CscMatrix: Mul<&'a X, Output = Vector>,
{
    fn minimize_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
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

#[expect(clippy::too_many_arguments)]
pub(super) fn iterate<B, S, J, H, X, V>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    search: B,
    mut objective: impl FnMut(&X) -> Result<V, String>,
    jacobian: impl FnMut(&X) -> Result<J, String>,
    hessian: impl FnMut(&X) -> Result<H, String>,
    update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    mut initial_guess: X,
    equality_constraint: EqualityConstraints,
    sparse: Option<SparseSolver>,
    lent: bool,
) -> Result<X, OptimizationError>
where
    B: Backtrack + Gradient<X, J, X, V>,
    V: Erase<Erased = Scalar>,
    S: Display,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    X: Jacobian + Solution,
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
            B::OBJECTIVE.then(|| verified(sparse)),
            indices,
        ),
        (indices, Rows::Dense(constraint_matrix, constraint_rhs)) => constrained(
            newton_raphson,
            search,
            |argument: &X| objective(argument).map(|value| *value.erase()),
            jacobian,
            hessian,
            update,
            initial_guess,
            sparse,
            &indices.unwrap_or_default(),
            constraint_matrix,
            constraint_rhs,
        ),
        (indices, Rows::Sparse(constraint_matrix, constraint_rhs)) => constrained(
            newton_raphson,
            search,
            |argument: &X| objective(argument).map(|value| *value.erase()),
            jacobian,
            hessian,
            update,
            initial_guess,
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
            )
        }
    }
    .map_err(|error| OptimizationError::upstream(error, newton_raphson))
}

#[expect(clippy::too_many_arguments)]
pub(super) fn iterate_linear<B, L, S, J, H, X, V>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: B,
    objective: impl FnMut(&X) -> Result<V, String>,
    jacobian: impl FnMut(&X) -> Result<J, String>,
    hessian: impl FnMut(&X) -> Result<H, String>,
    update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    mut initial_guess: X,
    equality_constraint: EqualityConstraints,
) -> Result<X, OptimizationError>
where
    B: Backtrack + Gradient<X, J, X, V>,
    V: Erase<Erased = Scalar>,
    L: LinearSolver<Tangent = H, Residual = J, Decrement = X>,
    S: Display,
    J: Jacobian,
    X: Jacobian + Solution,
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
            |tangent, fixed, residual| {
                Ok(newton_raphson
                    .linear_solver
                    .solve(tangent, fixed, residual)?)
            },
            None::<fn(H, &[bool], &[usize]) -> Option<bool>>,
            indices,
        ),
        _ => Err(OptimizationError::Intermediate(
            "This linear solver requires a fixed equality constraint.".to_string(),
        )),
    }
    .map_err(|error| OptimizationError::upstream(error, newton_raphson))
}

pub(super) fn direct<H, F, X>(
    sparse: Option<SparseSolver>,
) -> impl FnMut(H, &[usize], &F) -> Result<X, OptimizationError>
where
    H: Hessian,
    F: Jacobian,
    X: Jacobian + Solution,
{
    let mut factorization = None;
    move |hessian, fixed, residual| {
        let size = residual.size();
        let mut retained = vec![true; size];
        fixed.iter().for_each(|&index| retained[index] = false);
        let unmap: Vec<usize> = retained
            .iter()
            .enumerate()
            .filter_map(|(index, &keep)| keep.then_some(index))
            .collect();
        let mut flattened = Vector::zero(size);
        residual.fill_into(&mut flattened);
        let reduced: Vector = unmap.iter().map(|&index| flattened[index]).collect();
        let solved = if let Some(ref solver) = sparse {
            solver.solve(|i, j| hessian.entry(unmap[i], unmap[j]), &reduced)?
        } else {
            let factorization =
                factorization.get_or_insert_with(|| LuDecomposition::zero(unmap.len()));
            hessian
                .retain_from(&retained)
                .factorize_lu_into(factorization)?;
            let mut solved = Vector::zero(unmap.len());
            factorization.solve_into(&reduced, &mut solved);
            solved
        };
        flattened.iter_mut().for_each(|entry| *entry = 0.0);
        unmap
            .into_iter()
            .zip(solved)
            .for_each(|(index, entry)| flattened[index] = entry);
        Ok(X::from(flattened))
    }
}
