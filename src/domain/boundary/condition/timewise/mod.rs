#[cfg(test)]
mod test;

use super::ConstrainAt;
use crate::{
    math::{Quantity, optimize::EqualityConstraints},
    units::Time,
};

/// Constraints given as a function of time, such as rows that no other condition expresses.
///
/// A solver that steps through time holds what is prescribed relative to the initial guess at
/// that distance from the reference, as it does for any other condition.
pub struct Timewise<F>(pub F);

impl<L, F> ConstrainAt<L> for Timewise<F>
where
    F: Fn(Quantity<Time>) -> EqualityConstraints,
{
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        (self.0)(time)
    }
}
