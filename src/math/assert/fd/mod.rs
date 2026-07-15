use crate::math::Scalar;

/// Types that can report a finite-difference comparison error against themselves.
pub trait FiniteDifference {
    fn error_fd(&self, comparator: &Self, epsilon: Scalar) -> Option<(bool, usize)>;
}
