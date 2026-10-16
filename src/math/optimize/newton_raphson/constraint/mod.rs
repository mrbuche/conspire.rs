#[cfg(test)]
mod test;

use super::super::super::{Hessian, Matrix, Scalar, Tensor, Vector, sparse::CscMatrix};

/// The constraints of a bordered KKT system, with a row for each constraint.
pub(super) trait ConstraintMatrix {
    /// The entry of a constraint on a variable.
    fn at(&self, constraint: usize, variable: usize) -> Scalar;
    /// Visits the entries that may be nonzero.
    fn for_each_nonzero(&self, visit: impl FnMut(usize, usize, Scalar));
    /// The sum of the constraints weighted by their multipliers.
    fn weighted(&self, multipliers: &Vector) -> Vector;
}

impl ConstraintMatrix for Matrix {
    fn at(&self, constraint: usize, variable: usize) -> Scalar {
        self[constraint][variable]
    }
    fn weighted(&self, multipliers: &Vector) -> Vector {
        multipliers * self
    }
    fn for_each_nonzero(&self, mut visit: impl FnMut(usize, usize, Scalar)) {
        self.iter().enumerate().for_each(|(i, row)| {
            row.iter()
                .enumerate()
                .for_each(|(j, &value)| visit(i, j, value))
        })
    }
}

impl ConstraintMatrix for CscMatrix {
    fn at(&self, constraint: usize, variable: usize) -> Scalar {
        self.entry(constraint, variable)
    }
    fn weighted(&self, multipliers: &Vector) -> Vector {
        multipliers * self
    }
    fn for_each_nonzero(&self, mut visit: impl FnMut(usize, usize, Scalar)) {
        self.iter()
            .for_each(|(constraint, variable, &value)| visit(constraint, variable, value))
    }
}

pub(super) fn kkt_entry_sparse<H: Hessian, M: ConstraintMatrix>(
    hessian: &H,
    constraint_matrix: &M,
    num_variables: usize,
    row: usize,
    column: usize,
) -> Scalar {
    if row >= num_variables {
        -constraint_matrix.at(row - num_variables, column)
    } else if column >= num_variables {
        -constraint_matrix.at(column - num_variables, row)
    } else {
        hessian.entry(row, column)
    }
}
