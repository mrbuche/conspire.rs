use super::super::{Quantity, Scalar};
use crate::ABS_TOL;

/// Absolute error tolerances.
///
/// A constrained problem meets three quantities of different units, so one
/// number cannot serve them all. Which units those are is settled by the
/// problem rather than chosen here, so each is named where it is read back
/// rather than where it is set.
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    /// Absolute error tolerance on the constraint violation.
    pub constraint: Scalar,
    /// Absolute error tolerance on the residual.
    pub residual: Scalar,
    /// Absolute error tolerance on the slope the line search measures.
    pub slope: Scalar,
}

impl Tolerances {
    /// The tolerance on the constraint violation, as the unit it is met in.
    pub const fn constraint<U>(&self) -> Quantity<U> {
        Quantity::new(self.constraint)
    }
    /// The tolerance on the residual, as the unit it is met in.
    pub const fn residual<U>(&self) -> Quantity<U> {
        Quantity::new(self.residual)
    }
    /// The tolerance on the slope, as the unit it is met in.
    pub const fn slope<U>(&self) -> Quantity<U> {
        Quantity::new(self.slope)
    }
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            constraint: ABS_TOL,
            residual: ABS_TOL,
            slope: ABS_TOL,
        }
    }
}
