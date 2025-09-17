use crate::math::{Matrix, Vector};

/// Possible equality constraints.
#[derive(Clone)] // Clone is for passing from minimize to minimize_inner in fem/block/mod.rs/ElasticHyperviscousFiniteElementBlock
pub enum EqualityConstraint {
    /// Indices fixed at initial guess values.
    Fixed(Vec<usize>),
    /// Linear equality constraint.
    Linear(Matrix, Vector),
    /// No constraint.
    None,
}
