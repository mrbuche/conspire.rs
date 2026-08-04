use super::NewtonRaphson;

/// Possible strategies for problems split into global and local variables.
#[derive(Clone, Debug)]
pub enum SolveStrategy {
    /// Local variables converged at fixed global variables before each global
    /// step, by a solver of their own.
    ///
    /// The local problem is small, dense, and handed an exact tangent, so it is
    /// always Newton, but it need not be converged as tightly nor safeguarded
    /// the same way as the global one.
    Condensed(NewtonRaphson),
    /// Global and local variables stepped together, optionally eliminating the local block.
    Monolithic { elimination: bool },
}
