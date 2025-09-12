use crate::math::{Matrix, Vector};

/// Possible equality constraints.
#[derive(Clone)] // Clone is for passing from minimize to minimize_inner in fem/block/mod.rs/ElasticHyperviscousFiniteElementBlock
pub enum EqualityConstraint {
    // Fixed(Vec<usize>), give list of indices that say these DOFs are fixed at what they are in initial_guess
    Linear(Matrix, Vector),
    None,
}
