use super::{Direct, NewtonRaphson, NoSearch};
use std::fmt::Display;

/// Possible strategies for problems split into global and local variables.
#[derive(Clone, Debug)]
pub enum SolveStrategy<S: Display = NoSearch> {
    /// Local variables converged at fixed global variables before each global
    /// step, by a solver of their own.
    ///
    /// The local problem is small, dense, and handed an exact tangent, so it is
    /// always Newton, but it need not be converged as tightly nor safeguarded
    /// the same way as the global one.
    Condensed(NewtonRaphson<Direct, S>),
    /// Global and local variables stepped together, optionally eliminating the local block.
    Monolithic { elimination: bool },
}

impl SolveStrategy {
    /// Global and local variables stepped together, with nothing to infer a line search from.
    pub fn monolithic(elimination: bool) -> Self {
        Self::Monolithic { elimination }
    }
}
