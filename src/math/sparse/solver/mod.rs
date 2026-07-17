#[cfg(test)]
mod test;

use super::{
    SparseError,
    factor::{CscLdl, CscLu},
    matrix::CscMatrix,
};
use crate::math::{Scalar, Vector};
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

/// A sparse direct solver for repeated solves on a fixed sparsity pattern.
///
/// Cloning shares the cached factorization, whose pivot order and fill pattern
/// are reused across solves until a pivot degrades. Symmetric values use an
/// LDLᵀ factorization, falling back to LU otherwise. Symmetry is a caller-supplied
/// guarantee about the source values, not detected at runtime — a source whose
/// symmetry can vary between solves (e.g. a tangent that is only symmetric near
/// a particular configuration) must be declared asymmetric.
#[derive(Clone)]
pub struct SparseSolver {
    matrix: RefCell<CscMatrix>,
    ldl: Rc<RefCell<Option<CscLdl>>>,
    lu: Rc<RefCell<Option<CscLu>>>,
}

impl SparseSolver {
    pub fn from_pattern(num: usize, pattern: Vec<(usize, usize)>, symmetric: bool) -> Self {
        let matrix = CscMatrix::from_pattern(num, num, pattern);
        let ldl = if symmetric {
            matrix.ldl_symbolic().ok()
        } else {
            None
        };
        let lu = if ldl.is_none() {
            matrix.lu_symbolic().ok()
        } else {
            None
        };
        Self {
            matrix: RefCell::new(matrix),
            ldl: Rc::new(RefCell::new(ldl)),
            lu: Rc::new(RefCell::new(lu)),
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
            if let Some(cached) = ldl.as_mut()
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
