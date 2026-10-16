use crate::math::{Matrix, Scalar, Vector};

/// What an index of a prescribed constraint is held at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prescription {
    /// Its initial guess value.
    Initial,
    /// Its initial guess value plus an amount.
    Offset(Scalar),
    /// A value.
    Absolute(Scalar),
}

/// Possible equality constraints.
#[derive(Clone)]
pub enum EqualityConstraint {
    /// Indices fixed at initial guess values.
    Fixed(Vec<usize>),
    /// Indices held at what they are prescribed.
    Prescribed(Vec<(usize, Prescription)>),
    /// Linear equality constraint.
    Linear(Matrix, Vector),
    /// No constraint.
    None,
}
