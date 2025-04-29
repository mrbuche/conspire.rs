use crate::math::{Matrix, Vector};

/// Possible equality constraints.
pub enum EqualityConstraint {
    Linear(Matrix, Vector),
    None,
}
