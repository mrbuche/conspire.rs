use crate::math::{Matrix, Vector};

/// Possible equality constraints.
#[derive(Clone)] // Clone is for passing from minimize to minimize_inner in fem/block/mod.rs/ElasticHyperviscousFiniteElementBlock
pub enum EqualityConstraint {
    Linear(Matrix, Vector),
    None,
}
