use crate::math::{Tensor, Vector};

/// Residual modification to improve solver conditioning.
pub trait Precondition {
    fn apply(&self, residual: &Vector) -> Vector;
}

/// Available preconditioning methods.
pub enum Preconditioning {
    /// A diagonal matrix.
    Diagonal(Vector),
    /// No preconditioning.
    None,
}

impl Precondition for Preconditioning {
    fn apply(&self, residual: &Vector) -> Vector {
        match self {
            Self::None => residual.clone(),
            Self::Diagonal(diagonal) => residual
                .iter()
                .zip(diagonal.iter())
                .map(|(entry, scale)| entry / scale)
                .collect(),
        }
    }
}

impl<F> Precondition for F
where
    F: Fn(&Vector) -> Vector,
{
    fn apply(&self, residual: &Vector) -> Vector {
        self(residual)
    }
}
