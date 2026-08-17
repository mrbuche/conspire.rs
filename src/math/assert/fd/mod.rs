use crate::math::{Quantity, Scalar};

/// A step to perturb an entry by, in whatever unit that entry carries.
///
/// A finite difference divides by the step it took, so the step carries the
/// unit of the entry it steps rather than being the bare number a tolerance is,
/// and the quotient comes out in the unit the derivative is actually in.
pub const fn perturbation<U>(epsilon: Scalar) -> Quantity<U> {
    Quantity::new(epsilon)
}

/// Types that can report a finite-difference comparison error against themselves.
pub trait FiniteDifference {
    fn error_fd(&self, comparator: &Self, epsilon: Scalar) -> Option<(bool, usize)>;
}
