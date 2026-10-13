#[cfg(test)]
mod test;

mod constraint;
mod globalization;
mod linear;
mod nonlinear;
mod preconditioning;
mod strategy;
mod tolerance;

pub(crate) use constraint::Rows;
use constraint::UNSUPPORTED;
pub use constraint::{EqualityConstraint, EqualityConstraints, Prescription};
use globalization::BacktrackingLineSearch;
pub use globalization::{
    Armijo, Error, Goldstein, LineSearchError, NoSearch, ObjectiveMerit, Residual, ResidualMerit,
    TrustRegion, Wolfe,
};
pub use linear::{Direct, Krylov, KrylovError, KrylovMethod, LinearSolver};
pub use nonlinear::{
    BarzilaiBorwein, BarzilaiBorweinStep, Conjugacy, ConjugateGradient, Descent, GradientDescent,
    LimitedMemoryBFGS, NewtonRaphson, StepEstimator,
};
pub use preconditioning::{Jacobi, Precondition, Preconditioning, Scaled};
pub use strategy::SolveStrategy;
pub use tolerance::Tolerances;

#[cfg(feature = "constitutive")]
pub(crate) use nonlinear::{converged, limit_decrement};

use crate::{
    math::{
        Quantity, Scalar, Style, StyledError, Tensor, Vector,
        assert::AssertionError,
        matrix::square::SquareMatrixError,
        sparse::{CscMatrix, SparseError, SparseSolver},
        styled_error,
    },
    units::{UnitDiv, UnitMul},
};
use std::fmt::{Debug, Display};

/// The step size taking a decrement of type `D` to an increment of `X`.
pub type StepSize<D, X> = Quantity<<<X as Tensor>::Unit as UnitDiv<<D as Tensor>::Unit>>::Output>;

/// The objective a step along a gradient `J` of a solution `X` changes, where `A` is
/// the unit of the gradient contracted with itself.
pub type ObjectiveOf<J, X, A> =
    Quantity<<<<X as Tensor>::Unit as UnitDiv<<J as Tensor>::Unit>>::Output as UnitMul<A>>::Output>;

/// Root-finding algorithms, where a tangent of `()` is one the solver does not use.
pub trait RootFinding<F, J, X> {
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// Root-finding algorithms that hand out each increment before
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
pub trait RootFindingIncremental<F, J, X> {
    fn root_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// Optimization algorithms, where a Hessian of `()` is one the solver does not use.
pub trait Optimization<F, J, H, X> {
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// Optimization algorithms that hand out each increment before
/// applying it.
///
/// The counterpart of [`RootFindingIncremental`] for problems with an
/// energy to descend, and the increment is lent on the same terms.
///
/// What the line search measures is the energy of the whole state, eliminated
/// variables included. Each trial is offered through the same update, so the
/// eliminated variables are already standing where the trial puts them by the
/// time the energy there is asked for.
pub trait OptimizationIncremental<F, J, H, X> {
    #[expect(clippy::too_many_arguments)]
    fn minimize_incremental(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// Root-finding algorithms for problems split into global and local variables.
#[expect(clippy::too_many_arguments)]
pub trait RootFindingBlock<U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> {
    fn root_block(
        &self,
        residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
        residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
        tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
        initial_guess: (U, V),
        constraint_global: (CscMatrix, Vector),
        constraint_local: (CscMatrix, Vector),
        sparse: Option<SparseSolver>,
        strategy: SolveStrategy<impl ResidualMerit>,
    ) -> Result<(U, V), OptimizationError>;
}

/// Optimization algorithms for problems split into global and local variables.
#[expect(clippy::too_many_arguments)]
pub trait OptimizationBlock<F, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv> {
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
        strategy: SolveStrategy<impl ResidualMerit>,
    ) -> Result<(U, V), OptimizationError>;
}

fn no_objective<X, V>(_: &X) -> Result<V, String> {
    Err("A root has no objective.".to_string())
}

fn no_objective_blocks<U, V>(_: &U, _: &V) -> Result<Scalar, String> {
    Err("A root has no objective.".to_string())
}

/// Possible errors encountered during optimization.
pub enum OptimizationError {
    Intermediate(String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
    Upstream(String, String),
    SingularMatrix,
    UnsymmetricMatrix,
}

impl OptimizationError {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
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
            Self::UnsymmetricMatrix => format!("{h}Matrix is not symmetric."),
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
    fn from(error: SparseError) -> Self {
        match error {
            SparseError::Singular => Self::SingularMatrix,
            SparseError::Unsymmetric => Self::UnsymmetricMatrix,
        }
    }
}
