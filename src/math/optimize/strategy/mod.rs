/// Possible strategies for problems split into global and local variables.
#[derive(Clone, Copy, Debug)]
pub enum SolveStrategy {
    /// Local variables converged at fixed global variables before each global step.
    Condensed,
    /// Global and local variables stepped together, optionally eliminating the local block.
    Monolithic { elimination: bool },
}
