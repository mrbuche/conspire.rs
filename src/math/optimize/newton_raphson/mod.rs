#[cfg(test)]
mod test;

use super::{
    super::{
        Erase, Hessian, HessianBlock, Jacobian, LuDecomposition, Matrix, Quantity, Scalar,
        Solution, SquareMatrix, Tensor, Vector,
        sparse::{CscMatrix, SparseSolver},
        unit::{Dimensionless, UnitDiv},
    },
    BacktrackingLineSearch, EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock,
    FirstOrderRootFindingIncremental, LineSearch, LineSearchError, OptimizationError,
    SecondOrderOptimization, SecondOrderOptimizationBlock, SecondOrderOptimizationIncremental,
    SolveStrategy, TrustRegion,
};
use crate::ABS_TOL;
use crate::math::Norm;
use std::{
    fmt::{self, Debug, Formatter},
    ops::{Div, Mul},
};

/// The Newton-Raphson method.
#[derive(Clone)]
pub struct NewtonRaphson {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Norm type for error evaluation.
    pub error_norm: Norm,
    /// Line search algorithm.
    pub line_search: LineSearch,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// How far the step is trusted.
    pub trust_region: TrustRegion,
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
            "NewtonRaphson {{ abs_tol: {:?}, line_search: {}, max_steps: {:?}, trust_region: {:?} }}",
            self.abs_tol, self.line_search, self.max_steps, self.trust_region
        )
    }
}

impl Default for NewtonRaphson {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            error_norm: Norm::Chebyshev,
            line_search: LineSearch::None,
            max_steps: 25,
            trust_region: TrustRegion::None,
        }
    }
}

impl<F, J, X, E> FirstOrderRootFinding<F, J, X> for NewtonRaphson
where
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    F: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
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
        match match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                |_: &X, _: &Vector, _: Scalar, _: bool| Ok(()),
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                |_: &X, _: &Vector, _: Scalar, _: bool| Ok(()),
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
                sparse,
            ),
        } {
            Ok(solution) => Ok(solution),
            Err(error) => Err(OptimizationError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<F, J, X, E> FirstOrderRootFindingIncremental<F, J, X> for NewtonRaphson
where
    F: Jacobian,
    for<'a> &'a F: Div<J, Output = X>,
    J: Hessian,
    F: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        match match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                update,
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                |_: &X| panic!("No line search in root finding"),
                function,
                jacobian,
                update,
                initial_guess,
                sparse,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => unimplemented!(
                "An unconstrained solution has no chained vector to lend the increment through."
            ),
        } {
            Ok(solution) => Ok(solution),
            Err(error) => Err(OptimizationError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<J, H, X, E> SecondOrderOptimization<Scalar, J, H, X> for NewtonRaphson
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
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
                |_: &X, _: &Vector, _: Scalar, _: bool| Ok(()),
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                function,
                jacobian,
                hessian,
                |_: &X, _: &Vector, _: Scalar, _: bool| Ok(()),
                initial_guess,
                sparse,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => {
                unconstrained(self, function, jacobian, hessian, initial_guess, sparse)
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

impl<J, H, X, E> SecondOrderOptimizationIncremental<Scalar, J, H, X> for NewtonRaphson
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn minimize_incremental(
        &self,
        function: impl FnMut(&X) -> Result<Scalar, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
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
                update,
                initial_guess,
                sparse,
                indices,
            ),
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => constrained(
                self,
                function,
                jacobian,
                hessian,
                update,
                initial_guess,
                sparse,
                constraint_matrix,
                constraint_rhs,
            ),
            EqualityConstraint::None => unimplemented!(
                "An unconstrained solution has no chained vector to lend the increment through."
            ),
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
        match blocked(
            self,
            |_: &U, _: &V| panic!("No line search in root finding"),
            false,
            residual_global,
            residual_local,
            tangents,
            initial_guess,
            constraint_global,
            constraint_local,
            sparse,
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
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    fn minimize_block(
        &self,
        function: impl FnMut(&U, &V) -> Result<Scalar, String>,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (CscMatrix, Vector),
        constraint_local: (CscMatrix, Vector),
        sparse: Option<SparseSolver>,
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError> {
        match blocked(
            self,
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

fn violation<M, T>(constraint_matrix: &M, constraint_rhs: &Vector, variables: &T) -> Scalar
where
    for<'a> &'a M: Mul<&'a T, Output = Vector>,
{
    (constraint_rhs - constraint_matrix * variables)
        .iter()
        .map(|entry| entry.abs())
        .sum()
}

/// The entry of the whole Karush-Kuhn-Tucker matrix, ordered as the global
/// variables, their multipliers, the local variables, then theirs.
#[allow(clippy::too_many_arguments)]
fn kkt_entry<Kuu, Kvu, Kuv, Kvv>(
    row: usize,
    column: usize,
    num_global: usize,
    num_outer: usize,
    num_local: usize,
    tangent_uu: &Kuu,
    tangent_vu: &Kvu,
    tangent_uv: &Kuv,
    tangent_vv: &Kvv,
    constraint_matrix_global: &CscMatrix,
    constraint_matrix_local: &CscMatrix,
) -> Scalar
where
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
{
    let local = num_outer + num_local;
    let row_global = row < num_global;
    let row_local = (num_outer..local).contains(&row);
    let column_global = column < num_global;
    let column_local = (num_outer..local).contains(&column);
    if row_global && column_global {
        tangent_uu.entry(row, column)
    } else if row_global && column_local {
        tangent_uv.entry(row, column - num_outer)
    } else if row_local && column_global {
        tangent_vu.entry(row - num_outer, column)
    } else if row_local && column_local {
        tangent_vv.entry(row - num_outer, column - num_outer)
    } else if row_global && (num_global..num_outer).contains(&column) {
        -constraint_matrix_global.entry(column - num_global, row)
    } else if (num_global..num_outer).contains(&row) && column_global {
        -constraint_matrix_global.entry(row - num_global, column)
    } else if row_local && column >= local {
        -constraint_matrix_local.entry(column - local, row - num_outer)
    } else if row >= local && column_local {
        -constraint_matrix_local.entry(row - local, column - num_outer)
    } else {
        0.0
    }
}

/// Shortens the step until it lands somewhere the problem can be evaluated,
/// or gives up.
///
/// This asks nothing of a merit function, so it is the one line search root
/// finding can also take, and it says nothing about descent. Where a trial
/// point is, and what makes it reachable, is left to the formulation asking:
/// whatever was eliminated is stepped alongside, so the state to test is the
/// one everything arrives at together.
fn backtrack_errors(
    newton_raphson: &NewtonRaphson,
    mut reachable: impl FnMut(Scalar) -> bool,
    cut_back: Scalar,
    max_steps: usize,
) -> Result<Scalar, OptimizationError> {
    let mut trial_size = 1.0;
    for _ in 0..max_steps {
        if reachable(trial_size) {
            return Ok(trial_size);
        }
        trial_size *= cut_back
    }
    Err(OptimizationError::Upstream(
        format!(
            "{}",
            LineSearchError::MaximumStepsReached(
                format!("{:?}", newton_raphson.line_search),
                max_steps
            )
        ),
        format!("{newton_raphson:?}"),
    ))
}

/// Shortens the step until the variables move no further than the maximum.
///
/// Only the variables are measured, the multipliers being of another kind
/// entirely, but everything is scaled together so that the direction survives.
fn limit_decrement(newton_raphson: &NewtonRaphson, decrements: &mut [(&mut Vector, usize)]) {
    if let TrustRegion::Fixed { radius, norm } = newton_raphson.trust_region {
        let size = norm.over(
            decrements
                .iter()
                .flat_map(|(decrement, variables)| decrement.iter().take(*variables).copied()),
        );
        if size > radius {
            decrements
                .iter_mut()
                .for_each(|(decrement, _)| **decrement *= radius / size)
        }
    }
}

fn kkt_block<K>(
    tangent: &K,
    constraint_matrix: &CscMatrix,
    size: usize,
    block: &mut SquareMatrix,
    offset: usize,
) where
    K: HessianBlock,
{
    tangent.fill_into_block(block, offset, offset);
    constraint_matrix.iter().for_each(|(a, j, entry)| {
        block[offset + size + a][offset + j] = -entry;
        block[offset + j][offset + size + a] = -entry;
    })
}

fn kkt_residual<R, T>(
    residual: R,
    multipliers: &Vector,
    constraint_matrix: &CscMatrix,
    constraint_rhs: &Vector,
    variables: &T,
    chained: &mut Vector,
) where
    R: Jacobian,
    for<'a> &'a CscMatrix: Mul<&'a T, Output = Vector>,
{
    (residual - multipliers * constraint_matrix)
        .fill_into_chained(constraint_rhs - constraint_matrix * variables, chained)
}

/// Converges the local variables at fixed global ones.
///
/// The condensed strategy treats the local variables as a function of the
/// global ones, so anywhere the global variables are moved to, this is what
/// the local ones become.
#[allow(clippy::too_many_arguments)]
fn converge_local<U, V, Rv, Kuu, Kvu, Kuv, Kvv>(
    local_solver: &NewtonRaphson,
    residual_local: &mut impl FnMut(&U, &V) -> Result<Rv, String>,
    tangents: &mut impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
    global: &U,
    local: &mut V,
    multipliers_local: &mut Vector,
    constraint_matrix_local: &CscMatrix,
    constraint_rhs_local: &Vector,
    num_local: usize,
    update_inner: &mut Vector,
    tangent_inner: &mut SquareMatrix,
    factorization: &mut LuDecomposition,
) -> Result<(), OptimizationError>
where
    Rv: Jacobian,
    Kvv: HessianBlock,
    V: Solution,
    for<'a> &'a CscMatrix: Mul<&'a V, Output = Vector>,
{
    let mut local_steps = 0;
    loop {
        kkt_residual(
            residual_local(global, local)?,
            multipliers_local,
            constraint_matrix_local,
            constraint_rhs_local,
            local,
            update_inner,
        );
        if local_solver.error_norm.apply(update_inner) < local_solver.abs_tol
            || local_steps == local_solver.max_steps
        {
            return Ok(());
        }
        local_steps += 1;
        let (_, _, _, tangent) = tangents(global, local)?;
        kkt_block(
            &tangent,
            constraint_matrix_local,
            num_local,
            tangent_inner,
            0,
        );
        tangent_inner.factorize_lu_into(factorization)?;
        let mut decrement = factorization.solve(update_inner);
        limit_decrement(local_solver, &mut [(&mut decrement, num_local)]);
        local.decrement_from_chained(multipliers_local, &decrement)
    }
}

#[allow(clippy::too_many_arguments)]
fn blocked<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&U, &V) -> Result<Scalar, String>,
    minimizing: bool,
    mut residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
    mut residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
    mut tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
    initial_guess: (U, V),
    constraint_global: (CscMatrix, Vector),
    constraint_local: (CscMatrix, Vector),
    sparse: Option<SparseSolver>,
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
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
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
    let condensed = match strategy {
        SolveStrategy::Condensed(ref local_solver) => Some(local_solver),
        SolveStrategy::Monolithic { .. } => None,
    };
    if sparse.is_some() && eliminating {
        unimplemented!(
            "Eliminating the local block sparsely wants it held as the blocks it is, not as one matrix."
        )
    }
    let (inner, outer) = if eliminating {
        (num_inner, num_outer)
    } else {
        (0, 0)
    };
    let whole = if eliminating || sparse.is_some() {
        0
    } else {
        num_outer + num_inner
    };
    let mut column = Vector::zero(inner);
    let mut coupling_global = Matrix::zero(outer, inner);
    let mut coupling_local = Matrix::zero(inner, outer);
    let mut decrement_inner = Vector::zero(num_inner);
    let mut decrement_outer = Vector::zero(num_outer);
    let mut decrement_whole = Vector::zero(whole);
    let mut eliminated = vec![Vector::zero(inner); outer.min(num_global)];
    let mut factorization = LuDecomposition::zero(inner);
    let mut factorization_outer = LuDecomposition::zero(outer);
    let mut factorization_whole = LuDecomposition::zero(whole);
    let mut monolithic = SquareMatrix::zero(whole);
    let mut residual = Vector::zero(num_outer + num_inner);
    let mut tangent_inner = SquareMatrix::zero(inner);
    let mut tangent_outer = SquareMatrix::zero(outer);
    let mut update_inner = Vector::zero(num_inner);
    let mut update_outer = Vector::zero(num_outer);
    let mut steps = 0;
    loop {
        if let Some(local_solver) = condensed {
            converge_local(
                local_solver,
                &mut residual_local,
                &mut tangents,
                &global,
                &mut local,
                &mut multipliers_local,
                &constraint_matrix_local,
                &constraint_rhs_local,
                num_local,
                &mut update_inner,
                &mut tangent_inner,
                &mut factorization,
            )?
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
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok((global, local));
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        }
        steps += 1;
        let (tangent_uu, tangent_vu, tangent_uv, tangent_vv) = tangents(&global, &local)?;
        if eliminating {
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
            factorization.solve_into(&update_inner, &mut decrement_inner);
            (0..num_global).for_each(|i| {
                (0..num_local).for_each(|j| {
                    let coupling = coupling_global[i][j];
                    (0..num_global)
                        .for_each(|k| tangent_outer[i][k] -= coupling * eliminated[k][j]);
                    update_outer[i] -= coupling * decrement_inner[j]
                })
            });
            tangent_outer.factorize_lu_into(&mut factorization_outer)?;
            factorization_outer.solve_into(&update_outer, &mut decrement_outer);
            (0..num_global).for_each(|k| {
                decrement_inner
                    .iter_mut()
                    .zip(eliminated[k].iter())
                    .for_each(|(decrement_inner_i, eliminated_ki)| {
                        *decrement_inner_i -= eliminated_ki * decrement_outer[k]
                    })
            });
        } else {
            if sparse.is_none() {
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
            }
            if let Some(ref solver) = sparse {
                //
                // The block layout is the same either way, so the entry a
                // sparse solver asks for is read from whichever block holds it.
                //
                decrement_whole = solver.solve(
                    |i, j| {
                        kkt_entry(
                            i,
                            j,
                            num_global,
                            num_outer,
                            num_local,
                            &tangent_uu,
                            &tangent_vu,
                            &tangent_uv,
                            &tangent_vv,
                            &constraint_matrix_global,
                            &constraint_matrix_local,
                        )
                    },
                    &residual,
                )?
            } else {
                tangent_uv.fill_into_block(&mut monolithic, 0, num_outer);
                tangent_vu.fill_into_block(&mut monolithic, num_outer, 0);
                monolithic.factorize_lu_into(&mut factorization_whole)?;
                factorization_whole.solve_into(&residual, &mut decrement_whole)
            }
            decrement_whole
                .iter()
                .zip(decrement_outer.iter_mut().chain(decrement_inner.iter_mut()))
                .for_each(|(entry, decrement_i)| *decrement_i = *entry);
        }
        limit_decrement(
            newton_raphson,
            &mut [
                (&mut decrement_outer, num_global),
                (&mut decrement_inner, num_local),
            ],
        );
        let step_size = if matches!(newton_raphson.line_search, LineSearch::None) {
            1.0
        } else if !minimizing
            && let LineSearch::Error {
                cut_back,
                max_steps,
            } = &newton_raphson.line_search
        {
            //
            // Root finding has no merit function to backtrack against, so the
            // trial point is judged by whether the problem can be evaluated
            // there at all. Minimization keeps its merit function instead.
            //
            backtrack_errors(
                newton_raphson,
                |trial_size| {
                    let mut trial_global = global.clone();
                    let mut trial_local = local.clone();
                    let mut trial_multipliers_global = multipliers_global.clone();
                    let mut trial_multipliers_local = multipliers_local.clone();
                    trial_global.decrement_from_chained(
                        &mut trial_multipliers_global,
                        &(&decrement_outer * trial_size),
                    );
                    let reached = if let Some(local_solver) = condensed {
                        converge_local(
                            local_solver,
                            &mut residual_local,
                            &mut tangents,
                            &trial_global,
                            &mut trial_local,
                            &mut trial_multipliers_local,
                            &constraint_matrix_local,
                            &constraint_rhs_local,
                            num_local,
                            &mut update_inner,
                            &mut tangent_inner,
                            &mut factorization,
                        )
                        .is_ok()
                    } else {
                        trial_local.decrement_from_chained(
                            &mut trial_multipliers_local,
                            &(&decrement_inner * trial_size),
                        );
                        true
                    };
                    reached
                        && residual_global(&trial_global, &trial_local).is_ok()
                        && residual_local(&trial_global, &trial_local).is_ok()
                        && tangents(&trial_global, &trial_local).is_ok()
                },
                *cut_back,
                *max_steps,
            )?
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
                            &(&decrement_outer * step),
                        );
                        //
                        // Condensed makes the local variables a function of the
                        // global ones, so a trial point is where they solve to,
                        // not where the increment predicted they would.
                        //
                        if let Some(local_solver) = condensed {
                            converge_local(
                                local_solver,
                                &mut residual_local,
                                &mut tangents,
                                &trial_global,
                                &mut trial_local,
                                &mut trial_multipliers_local,
                                &constraint_matrix_local,
                                &constraint_rhs_local,
                                num_local,
                                &mut update_inner,
                                &mut tangent_inner,
                                &mut factorization,
                            )
                            .map_err(|error| format!("{error}"))?
                        } else {
                            trial_local.decrement_from_chained(
                                &mut trial_multipliers_local,
                                &(&decrement_inner * step),
                            )
                        }
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
            global.decrement_from_chained(&mut multipliers_global, &decrement_outer);
            local.decrement_from_chained(&mut multipliers_local, &decrement_inner)
        } else {
            global.decrement_from_chained(&mut multipliers_global, &(&decrement_outer * step_size));
            local.decrement_from_chained(&mut multipliers_local, &(&decrement_inner * step_size))
        }
    }
}

fn unconstrained<J, H, X, E>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
{
    let mut decrement;
    let mut flattened = Vector::zero(if sparse.is_none() {
        0
    } else {
        initial_guess.size()
    });
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        } else {
            steps += 1;
            decrement = if let Some(ref solver) = sparse {
                let hess = hessian(&solution)?;
                residual.fill_into(&mut flattened);
                X::from(solver.solve(|i, j| hess.entry(i, j), &flattened)?)
            } else {
                &residual / hessian(&solution)?
            };
            if let TrustRegion::Fixed { radius, norm } = newton_raphson.trust_region {
                let size = norm.apply(&decrement);
                if size > radius {
                    decrement *= radius / size
                }
            }
            step_size = newton_raphson.backtracking_line_search::<X, E>(
                |trial: &X, _: Scalar| function(trial),
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
}

#[allow(clippy::too_many_arguments)]
fn constrained_fixed<J, H, X, E>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    mut update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    H: Hessian,
    J: Jacobian,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitDiv<<X as Tensor>::Unit, Output = Dimensionless>,
    for<'a> &'a X: Mul<Quantity<Dimensionless>, Output = X> + Mul<Scalar, Output = X>,
{
    let mut applied = Vector::zero(initial_guess.size());
    let mut retained = vec![true; initial_guess.size()];
    indices.iter().for_each(|&index| retained[index] = false);
    let unmap: Vec<usize> = retained
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index))
        .collect();
    let mut decrement = Vector::zero(unmap.len());
    let mut factorization = LuDecomposition::zero(if sparse.is_none() { unmap.len() } else { 0 });
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?.retain_from(&retained);
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        } else if let Some(ref solver) = sparse {
            let hess = hessian(&solution)?;
            decrement = solver.solve(|i, j| hess.entry(unmap[i], unmap[j]), &residual)?
        } else {
            hessian(&solution)?
                .retain_from(&retained)
                .factorize_lu_into(&mut factorization)?;
            factorization.solve_into(&residual, &mut decrement)
        }
        steps += 1;
        limit_decrement(newton_raphson, &mut [(&mut decrement, unmap.len())]);
        //
        // Spread over the variables it belongs to before anything shortens it,
        // so that whatever was eliminated is offered the whole direction and
        // the fraction of it being taken, rather than a direction of its own.
        //
        applied.iter_mut().for_each(|entry| *entry = 0.0);
        unmap
            .iter()
            .zip(decrement.iter())
            .for_each(|(&index, decrement_a)| applied[index] = *decrement_a);
        step_size = if matches!(newton_raphson.line_search, LineSearch::None) {
            1.0
        } else if let LineSearch::Error {
            cut_back,
            max_steps,
        } = &newton_raphson.line_search
        {
            backtrack_errors(
                newton_raphson,
                |trial_size| {
                    let mut trial = solution.clone();
                    trial.decrement_from_retained(&retained, &(&decrement * trial_size));
                    update(&solution, &applied, trial_size, false).is_ok()
                        && jacobian(&trial).is_ok()
                },
                *cut_back,
                *max_steps,
            )?
        } else {
            let jac = jacobian(&solution)?;
            let mut decrement_full = &solution * 0.0;
            decrement_full.decrement_from_retained(&retained, &decrement);
            decrement_full *= -1.0;
            newton_raphson.backtracking_line_search::<X, E>(
                |trial: &X, step: Scalar| {
                    update(&solution, &applied, step, false)?;
                    function(trial)
                },
                &mut jacobian,
                &solution,
                &jac,
                &decrement_full,
                1.0,
            )?
        };
        update(&solution, &applied, step_size, true)?;
        if step_size != 1.0 {
            decrement *= step_size
        }
        solution.decrement_from_retained(&retained, &decrement)
    }
}

#[allow(clippy::too_many_arguments)]
fn constrained<J, H, X>(
    newton_raphson: &NewtonRaphson,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    mut update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
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
    let mut penalty = 0.0 as Scalar;
    let num_variables = initial_guess.size();
    let mut applied = Vector::zero(num_variables);
    let num_constraints = constraint_rhs.len();
    let num_total = num_variables + num_constraints;
    let mut decrement = Vector::zero(num_total);
    let mut factorization = LuDecomposition::zero(if sparse.is_none() { num_total } else { 0 });
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
    let mut steps = 0;
    loop {
        (jacobian(&solution)? - &multipliers * &constraint_matrix).fill_into_chained(
            &constraint_rhs - &constraint_matrix * &solution,
            &mut residual,
        );
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol {
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
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
            tangent.factorize_lu_into(&mut factorization)?;
            factorization.solve_into(&residual, &mut decrement)
        }
        steps += 1;
        limit_decrement(newton_raphson, &mut [(&mut decrement, num_variables)]);
        //
        // Only the variables are lent out, the multipliers chained onto the end
        // of the decrement being of another kind entirely.
        //
        applied
            .iter_mut()
            .zip(decrement.iter())
            .for_each(|(applied_i, decrement_i)| *applied_i = *decrement_i);
        let step_size = if matches!(newton_raphson.line_search, LineSearch::None) {
            1.0
        } else if let LineSearch::Error {
            cut_back,
            max_steps,
        } = &newton_raphson.line_search
        {
            backtrack_errors(
                newton_raphson,
                |trial_size| {
                    let mut trial = solution.clone();
                    let mut trial_multipliers = multipliers.clone();
                    trial
                        .decrement_from_chained(&mut trial_multipliers, &(&decrement * trial_size));
                    update(&solution, &applied, trial_size, false).is_ok()
                        && jacobian(&trial).is_ok()
                },
                *cut_back,
                *max_steps,
            )?
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
            update(&solution, &applied, 0.0, false)?;
            let value = function(&solution)? + violated;
            if slope < newton_raphson.abs_tol {
                1.0
            } else {
                match newton_raphson.line_search.backtrack_merit(
                    |step| {
                        let mut trial = solution.clone();
                        let mut trial_multipliers = multipliers.clone();
                        trial.decrement_from_chained(&mut trial_multipliers, &(&decrement * step));
                        update(&solution, &applied, step, false)?;
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
        //
        // The increment is lent out whole, before it is applied and before it
        // is shortened, so that the eliminated variables take the same fraction
        // of their own direction as the retained ones take of theirs.
        //
        update(&solution, &applied, step_size, true)?;
        if step_size != 1.0 {
            decrement *= step_size
        }
        solution.decrement_from_chained(&mut multipliers, &decrement)
    }
}
