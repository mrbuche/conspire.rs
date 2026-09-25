use crate::math::Vector;

/// The built-in linear solver: a sparse factorization when the problem offers
/// one, a dense factorization otherwise.
#[derive(Clone, Copy, Debug, Default)]
pub struct Direct;

/// A way of solving the linear system a Newton step is made of.
///
/// The tangent is the one this solver can work from, as the problem hands it
/// out. Choosing a solver for a problem that cannot supply that tangent is
/// refused at compile time.
pub trait LinearSolver {
    /// The tangent this solver works from.
    type Tangent;
    /// The decrement over the retained variables, given the tangent at the
    /// current state and the residual over those variables.
    fn solve(
        &self,
        tangent: Self::Tangent,
        retained: &[usize],
        residual: &Vector,
    ) -> Result<Vector, String>;
}
