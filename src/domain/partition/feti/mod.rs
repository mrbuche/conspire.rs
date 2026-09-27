#![allow(dead_code)]

pub(crate) mod dual;
pub(crate) mod dual_primal;
pub(crate) mod interface;
pub(crate) mod parallel;
pub(crate) mod pcg;
pub(crate) mod subdomain;
#[cfg(test)]
mod test;

/// Most threads any parallel stage of the solve uses, setup and PCG alike.
///
/// Defaults to serial: internal parallelism a caller didn't ask for can
/// oversubscribe an already-parallel outer context, and `dual::dual_reduce`'s
/// chunking makes the exact (non-associative) floating-point result depend
/// on this count, so a machine-dependent default would make the same solve
/// round differently on different hardware. Raise it only once a real
/// multi-subdomain problem has been benchmarked against it.
pub(crate) const THREADS: usize = 1;
