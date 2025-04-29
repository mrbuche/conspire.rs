pub mod linear;

use linear::LinearEqualityConstraint;

// linear constraints are still more general than the Dirichlet constraints being used
// consider that in the future and maybe make an enum for Linear too?

/// Possible equality constraints.
pub enum EqualityConstraint<T> {
    Linear(T),
    None,
}
