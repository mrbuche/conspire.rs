//! Whether a run of explicit integration checks that its time step is stable.

use crate::math::Scalar;

/// Takes the steps that are given, however large they are.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;

/// Checks each step against the fastest time scale of the elements, and a step above the
/// fraction `safety` of the stability limit is an error.
///
/// The estimate assembles the element tangents, and is refreshed every `interval` evaluations
/// of the bound, which for the integrators here is every step. It is that of the lumped masses
/// and heat capacities, which consistent ones exceed, so only a run with the lumped ones can be
/// checked.
#[derive(Clone, Copy, Debug)]
pub struct Checked {
    pub safety: Scalar,
    pub interval: usize,
}
