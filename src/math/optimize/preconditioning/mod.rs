use crate::math::{Tensor, Vector};

/// Residual modification to improve solver conditioning.
pub trait Precondition<F = Vector, X = Vector> {
    fn apply(&self, residual: &F) -> X;
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

impl<G, F, X> Precondition<F, X> for G
where
    G: Fn(&F) -> X,
{
    fn apply(&self, residual: &F) -> X {
        self(residual)
    }
}
