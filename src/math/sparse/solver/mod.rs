#[cfg(test)]
mod test;

use super::{SparseError, lu::CscLu, matrix::CscMatrix};
use crate::math::{Scalar, Vector};
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

/// A sparse direct solver for repeated solves on a fixed sparsity pattern.
/// Cloning shares the cached factorization, whose pivot order and fill pattern
/// are reused across solves until a pivot degrades.
#[derive(Clone)]
pub struct SparseSolver {
    matrix: RefCell<CscMatrix>,
    lu: Rc<RefCell<Option<CscLu>>>,
}

impl SparseSolver {
    pub fn from_pattern(num: usize, pattern: Vec<(usize, usize)>) -> Self {
        Self {
            matrix: RefCell::new(CscMatrix::from_pattern(num, num, pattern)),
            lu: Rc::new(RefCell::new(None)),
        }
    }
    /// The nonzero (row, column) positions this structure was built from.
    pub fn pattern(&self) -> Ref<'_, [(usize, usize)]> {
        Ref::map(self.matrix.borrow(), |matrix| matrix.pattern())
    }
    /// Solve a system of linear equations with values from a source,
    /// refactoring the cached factorization when possible.
    pub fn solve(
        &self,
        source: impl FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        let mut matrix = self.matrix.borrow_mut();
        matrix.fill(source);
        let mut lu = self.lu.borrow_mut();
        if let Some(cached) = lu.as_mut()
            && cached.refactor(&matrix).is_ok()
        {
            return Ok(cached.solve(b));
        }
        let mut symbolic = matrix.lu_symbolic();
        let fresh = if symbolic.refactor(&matrix).is_ok() {
            symbolic
        } else {
            matrix.lu_amd()?
        };
        let solution = fresh.solve(b);
        *lu = Some(fresh);
        Ok(solution)
    }
}
