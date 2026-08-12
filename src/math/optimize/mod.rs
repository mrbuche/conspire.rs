#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod line_search;
mod newton_raphson;
mod strategy;
mod trust_region;

pub use constraint::EqualityConstraint;
pub use gradient_descent::GradientDescent;
pub use line_search::{LineSearch, LineSearchError};
pub use newton_raphson::NewtonRaphson;
pub use strategy::SolveStrategy;
pub use trust_region::TrustRegion;

use crate::math::unit::UnitDiv;
use crate::math::{
    Erase, Jacobian, Quantity, Scalar, Solution, Style, StyledError, Tensor, Vector,
    assert::AssertionError,
    matrix::square::SquareMatrixError,
    sparse::{CscMatrix, SparseError, SparseSolver},
    styled_error,
};
use std::{fmt::Debug, ops::Mul};

/// The step size that takes a decrement of type `D` to an increment of `X`.
///
/// Its unit is that of the unknown over that of the decrement, so that the two
/// need not carry the same one and neither has to give theirs up.
pub type StepSize<D, X> = Quantity<<<X as Tensor>::Unit as UnitDiv<<D as Tensor>::Unit>>::Output>;

/// Zeroth-order root-finding algorithms.
///
/// `F` is the residual and `X` the unknown. The step size carries the unit of
/// the unknown over that of the residual, which is read off the two rather than
/// passed in.
pub trait ZerothOrderRootFinding<F, X> {
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// First-order root-finding algorithms.
pub trait FirstOrderRootFinding<F, J, X> {
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// First-order root-finding algorithms that hand out each increment before
/// applying it.
///
/// The solver keeps the iteration; the increment is only lent to the caller so
/// that whatever was eliminated from the system can be carried along with it.
///
/// The increment is lent whole, with the step it is about to be scaled by
/// alongside. Elimination solves one direction for the eliminated variables and
/// the retained ones together, so shortening the step has to shorten both by
/// the same amount, exactly as it would if nothing had been eliminated. Handing
/// over the shortened increment instead would invite a fresh solve against it,
/// which is a different direction rather than less of the same one.
///
/// A step is offered before it is taken. The caller is asked to report whether
/// the state it arrives at is admissible, and only later told to keep it.
pub trait FirstOrderRootFindingIncremental<F, J, X> {
    fn root_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// First-order optimization algorithms.
///
/// `F` is the objective, `J` its gradient, and `X` the unknown. The step size
/// carries the unit of the unknown squared over that of the objective, since the
/// gradient carries the objective over the unknown — the unit an inverse Hessian
/// carries, which is what a step size stands in for. It is read off `J` and `X`
/// rather than passed in.
///
/// The objective carries its unit in and gives it up inside, where the merit a
/// line search compares adds a constraint violation to it. What the gradient
/// contracted with the unknown carries is what the objective carries, since a
/// line search takes one against the other, and an implementation says so as a
/// bound rather than trusting the two to agree.
pub trait FirstOrderOptimization<F, J, X> {
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// Second-order optimization algorithms.
///
/// The objective is carried and spent as it is in [`FirstOrderOptimization`].
pub trait SecondOrderOptimization<F, J, H, X> {
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// Second-order optimization algorithms that hand out each increment before
/// applying it.
///
/// The counterpart of [`FirstOrderRootFindingIncremental`] for problems with an
/// energy to descend, and the increment is lent on the same terms.
///
/// What the line search measures is the energy of the whole state, eliminated
/// variables included. Each trial is offered through the same update, so the
/// eliminated variables are already standing where the trial puts them by the
/// time the energy there is asked for.
pub trait SecondOrderOptimizationIncremental<F, J, H, X> {
    #[allow(clippy::too_many_arguments)]
    fn minimize_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// First-order root-finding algorithms for problems split into global and local variables.
#[allow(clippy::too_many_arguments)]
pub trait FirstOrderRootFindingBlock<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> {
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
    ) -> Result<(U, V), OptimizationError>;
}

/// Second-order optimization algorithms for problems split into global and local variables.
#[allow(clippy::too_many_arguments)]
pub trait SecondOrderOptimizationBlock<F, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> {
    fn minimize_block(
        &self,
        function: impl FnMut(&U, &V) -> Result<F, String>,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (CscMatrix, Vector),
        constraint_local: (CscMatrix, Vector),
        sparse: Option<SparseSolver>,
        strategy: SolveStrategy,
    ) -> Result<(U, V), OptimizationError>;
}

trait BacktrackingLineSearch<J, X>
where
    Self: Debug,
{
    fn backtracking_line_search<D, E>(
        &self,
        mut function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, OptimizationError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E> + Tensor,
        E: Tensor,
        X: Solution,
        <X as Tensor>::Unit: UnitDiv<<D as Tensor>::Unit>,
        for<'a> &'a D: Mul<StepSize<D, X>, Output = X>,
    {
        if matches!(self.get_line_search(), LineSearch::None) {
            Ok(step_size)
        } else {
            match self.get_line_search().backtrack(
                &mut function,
                &mut jacobian,
                argument,
                jacobian0,
                decrement,
                step_size,
            ) {
                Ok(step_size) => Ok(step_size),
                Err(error) => Err(OptimizationError::Upstream(
                    format!("{error}"),
                    format!("{self:?}"),
                )),
            }
        }
    }
    fn get_line_search(&self) -> &LineSearch;
}

/// Possible errors encountered during optimization.
pub enum OptimizationError {
    Intermediate(String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
    Upstream(String, String),
    SingularMatrix,
}

impl From<String> for OptimizationError {
    fn from(error: String) -> Self {
        Self::Intermediate(error)
    }
}

impl StyledError for OptimizationError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::Intermediate(message) => message.to_string(),
            Self::MaximumStepsReached(steps, solver) => format!(
                "{h}Maximum number of steps ({steps}) reached.{c}\n\
                In solver: {solver}."
            ),
            Self::NotMinimum(solution, solver) => format!(
                "{h}The obtained solution is not a minimum.{c}\n\
                For solution: {solution}.\n\
                In solver: {solver}."
            ),
            Self::SingularMatrix => format!("{h}Matrix is singular."),
            Self::Upstream(error, solver) => format!(
                "{error}{c}\n\
                In solver: {solver}."
            ),
        }
    }
}

styled_error!(OptimizationError);

impl From<OptimizationError> for String {
    fn from(error: OptimizationError) -> Self {
        error.to_string()
    }
}

impl From<OptimizationError> for AssertionError {
    fn from(error: OptimizationError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<SquareMatrixError> for OptimizationError {
    fn from(_error: SquareMatrixError) -> Self {
        Self::SingularMatrix
    }
}

impl From<SparseError> for OptimizationError {
    fn from(_error: SparseError) -> Self {
        Self::SingularMatrix
    }
}
