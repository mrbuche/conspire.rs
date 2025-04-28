pub mod linear;

use linear::LinearEqualityConstraint;

/// Possible equality constraints.
pub enum EqualityConstraint<T> {
    Linear(T),
    None,
}
