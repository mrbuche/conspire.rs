mod factor;
mod matrix;
mod solver;

pub use factor::{CscLdl, CscLu};
pub use matrix::CscMatrix;
pub use solver::SparseSolver;

use crate::math::assert::AssertionError;

/// Possible errors for sparse data types.
#[derive(Debug, PartialEq)]
pub enum SparseError {
    Singular,
    Unsymmetric,
}

impl From<SparseError> for AssertionError {
    fn from(error: SparseError) -> Self {
        Self {
            message: format!("{error:?}"),
        }
    }
}
