mod amd;
mod lu;
mod matrix;
mod solver;

pub use lu::CscLu;
pub use matrix::CscMatrix;
pub use solver::SparseSolver;

/// Possible errors for sparse data types.
#[derive(Debug, PartialEq)]
pub enum SparseError {
    Singular,
}

impl From<SparseError> for crate::math::TestError {
    fn from(error: SparseError) -> Self {
        Self {
            message: format!("{error:?}"),
        }
    }
}
