#[cfg(test)]
mod test;

mod backtrack;
mod blocked;
mod constrained;
mod constraint;
mod convergence;
mod fixed;
mod kkt;
mod penalty;
mod solve;
mod unconstrained;
mod verify;

#[cfg(feature = "constitutive")]
pub(crate) use convergence::{converged, limit_decrement};

use super::{
    super::Scalar, BacktrackingLineSearch, Direct, NoSearch, ObjectiveMerit, Tolerances,
    TrustRegion,
};
use crate::math::Norm;
use std::fmt::{self, Debug, Display, Formatter};

/// The Newton-Raphson method.
#[derive(Clone)]
pub struct NewtonRaphson<L = Direct, S = NoSearch> {
    /// Absolute error tolerances.
    pub abs_tol: Tolerances,
    /// Norm type for error evaluation.
    pub error_norm: Norm,
    /// Line search algorithm.
    pub line_search: S,
    /// Linear solver for each step.
    pub linear_solver: L,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance.
    pub rel_tol: Option<Scalar>,
    /// How far the step is trusted.
    pub trust_region: TrustRegion,
}

impl<L, S, J, X> BacktrackingLineSearch<J, X> for NewtonRaphson<L, S>
where
    S: ObjectiveMerit,
{
    type Search = S;
    fn line_search(&self) -> &S {
        &self.line_search
    }
}

impl<L, S> Debug for NewtonRaphson<L, S>
where
    S: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NewtonRaphson {{ abs_tol: {:?}, line_search: {}, max_steps: {:?}, rel_tol: {:?}, trust_region: {:?} }}",
            self.abs_tol, self.line_search, self.max_steps, self.rel_tol, self.trust_region
        )
    }
}

impl NewtonRaphson {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::with(Direct)
    }
}

impl<L> Default for NewtonRaphson<L>
where
    L: Default,
{
    fn default() -> Self {
        Self::with(L::default())
    }
}

impl<L, S> NewtonRaphson<L, S> {
    /// The same solver, searching along each step with another line search.
    pub fn with_line_search<T>(self, line_search: T) -> NewtonRaphson<L, T> {
        NewtonRaphson {
            abs_tol: self.abs_tol,
            error_norm: self.error_norm,
            line_search,
            linear_solver: self.linear_solver,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            trust_region: self.trust_region,
        }
    }
}

impl<L> NewtonRaphson<L> {
    fn with(linear_solver: L) -> Self {
        Self {
            abs_tol: Tolerances::default(),
            error_norm: Norm::Chebyshev,
            line_search: NoSearch,
            linear_solver,
            max_steps: 25,
            rel_tol: None,
            trust_region: TrustRegion::None,
        }
    }
}
