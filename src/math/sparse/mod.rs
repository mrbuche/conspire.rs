mod amd;
mod lu;
mod matrix;

pub use lu::CscLu;
pub use matrix::CscMatrix;

/// Possible errors for sparse data types.
#[derive(Debug, PartialEq)]
pub enum SparseError {
    Singular,
}
