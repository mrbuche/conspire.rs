use super::super::sparse::SparseSolver;

/// How the linear system at each step is solved.
#[derive(Clone, Default)]
pub enum LinearSolver {
    /// A dense direct factorization, formed fresh from the whole tangent.
    #[default]
    Dense,
    /// A sparse direct factorization, whose pivot order and fill pattern are
    /// reused across solves.
    Sparse(SparseSolver),
}

impl LinearSolver {
    /// Whether the tangent is formed dense and factorized whole.
    ///
    /// Only the dense solver needs somewhere to put a square matrix, so this is
    /// what says whether those buffers are worth the room.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }
}
