#[cfg(test)]
mod test;

use super::{
    SparseError,
    factor::{CscLdl, CscLu},
    matrix::CscMatrix,
};
use crate::math::{Scalar, Vector};
use std::{
    cell::{Cell, Ref, RefCell},
    rc::Rc,
};

/// A sparse direct solver for repeated solves on a fixed sparsity pattern.
/// Cloning shares the cached factorization, whose pivot order and fill pattern
/// are reused across solves until a pivot degrades. Symmetric values use an
/// LDLᵀ factorization, falling back to LU otherwise.
#[derive(Clone)]
pub struct SparseSolver {
    matrix: RefCell<CscMatrix>,
    ldl: Rc<RefCell<Option<CscLdl>>>,
    lu: Rc<RefCell<Option<CscLu>>>,
    symmetric: Rc<Cell<Option<bool>>>,
}

impl SparseSolver {
    pub fn from_pattern(num: usize, pattern: Vec<(usize, usize)>) -> Self {
        let matrix = CscMatrix::from_pattern(num, num, pattern);
        let ldl = matrix.ldl_symbolic().ok();
        let lu = if ldl.is_none() {
            matrix.lu_symbolic().ok()
        } else {
            None
        };
        Self {
            matrix: RefCell::new(matrix),
            ldl: Rc::new(RefCell::new(ldl)),
            lu: Rc::new(RefCell::new(lu)),
            symmetric: Rc::new(Cell::new(None)),
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
        let mut ldl = self.ldl.borrow_mut();
        if ldl.is_some() {
            let symmetric = self.symmetric.get().unwrap_or_else(|| {
                let symmetric = matrix.symmetric();
                self.symmetric.set(Some(symmetric));
                symmetric
            });
            if symmetric
                && let Some(cached) = ldl.as_mut()
                && cached.refactor(&matrix).is_ok()
            {
                return Ok(cached.solve(b));
            }
            *ldl = None;
        }
        drop(ldl);
        let mut lu = self.lu.borrow_mut();
        if let Some(cached) = lu.as_mut()
            && cached.refactor(&matrix).is_ok()
        {
            return Ok(cached.solve(b));
        }
        let fresh = match matrix.lu_symbolic() {
            Ok(mut symbolic) => match symbolic.refactor(&matrix) {
                Ok(()) => symbolic,
                Err(_) => matrix.lu_amd()?,
            },
            Err(_) => matrix.lu_amd()?,
        };
        let solution = fresh.solve(b);
        *lu = Some(fresh);
        Ok(solution)
    }
}
