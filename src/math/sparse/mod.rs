mod factor;
mod matrix;
mod solver;

pub use factor::{CscLdl, CscLu};
pub use matrix::CscMatrix;
pub use solver::SparseSolver;

use crate::math::{Style, StyledError, assert::AssertionError, styled_error};

/// Possible errors for sparse data types.
#[derive(PartialEq)]
pub enum SparseError {
    Singular,
    Unsymmetric,
}

impl StyledError for SparseError {
    fn message(&self, style: &Style) -> String {
        let h = style.headline;
        match self {
            Self::Singular => format!("{h}Matrix is singular."),
            Self::Unsymmetric => format!("{h}Matrix is not symmetric."),
        }
    }
}

styled_error!(SparseError);

impl From<SparseError> for AssertionError {
    fn from(error: SparseError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
