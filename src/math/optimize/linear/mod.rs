mod krylov;

pub use krylov::{Krylov, KrylovError, KrylovMethod};

/// The built-in direct linear solver.
///
/// A sparse factorization when the problem
/// offers one, a dense factorization otherwise.
#[derive(Clone, Copy, Debug, Default)]
pub struct Direct;

/// A way of solving the linear system a Newton step is made of.
pub trait LinearSolver {
    /// The tangent this solver works from.
    type Tangent;
    /// The residual this solver is given.
    type Residual;
    /// The decrement this solver finds.
    type Decrement;
    /// The decrement of the solution, given the tangent at the current state,
    /// the entries of the solution that are held fixed and the residual.
    ///
    /// The residual has no value where the solution is held fixed, and nor
    /// does the decrement.
    fn solve(
        &self,
        tangent: Self::Tangent,
        fixed: &[usize],
        residual: &Self::Residual,
    ) -> Result<Self::Decrement, String>;
}
