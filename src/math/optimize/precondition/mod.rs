use crate::math::{Tensor, Vector};

/// Something a residual can be put through on its way to becoming a
/// direction. Implemented for `Preconditioning`'s built-in choices and, via
/// the blanket impl below, for any bare closure — an operator-shaped
/// preconditioner (itself another matrix-free reduction, such as FETI's
/// lumped preconditioner) needs no variant of its own here.
pub trait Precondition {
    fn apply(&self, residual: &Vector) -> Vector;
}

/// A preconditioner already built from whatever the caller's operator is.
///
/// Ported (trimmed) from the unmerged `line-search` branch, where this was
/// built from an assembled `Hessian`; here the caller builds it directly, since
/// an operator given only as a matvec closure has no entries to read one from.
pub enum Preconditioning {
    /// Nothing to put the residual through.
    None,
    /// A diagonal to divide it by.
    Diagonal(Vector),
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
