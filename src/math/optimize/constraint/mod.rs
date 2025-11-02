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

// Would still be `Linear(Matrix, Vector)` when passed to solver within `root_inner`
// but needs to be more like `Linear(Matrix, fn(t)->Vector)` when passed to `root`
// so maybe add another variant and impl method to evaluate it at `t` to return `Linear(Matrix, Vector)`
//
//  LinearTimeDependent(Matrix, fn(&Scalar) -> Vector),
//
// Might make more sense to have a separate enum for boundary conditions now,
// both in constitutive/ and in fem/ for all the model types,
// since they are more general than just equality constraints.
