#[cfg(test)]
mod test;

use super::super::super::{
    Hessian, Jacobian, Matrix, Scalar, SquareMatrix, Tensor, Vector, sparse::CscMatrix,
};

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

/// The variables of a bordered system held where they are by decoupling them.
///
/// Their rows and columns are those of the identity, their residual is zero, and the
/// constraints no longer see them move, while still seeing the values they hold.
pub(super) struct Eliminated {
    indices: Vec<usize>,
    mask: Vec<bool>,
}

impl Eliminated {
    pub(super) fn new(indices: &[usize], num_variables: usize) -> Self {
        let mut mask = vec![false; num_variables];
        indices.iter().for_each(|&index| mask[index] = true);
        Self {
            indices: indices.to_vec(),
            mask,
        }
    }
    fn contains(&self, index: usize) -> bool {
        self.mask.get(index).copied().unwrap_or(false)
    }
    pub(super) fn hold<J: Jacobian>(&self, mut jacobian: J) -> J {
        if !self.indices.is_empty() {
            jacobian.zero_out(&self.indices)
        }
        jacobian
    }
    pub(super) fn decouple(&self, tangent: &mut SquareMatrix) {
        if tangent.len() == 0 {
            return;
        }
        self.indices.iter().for_each(|&index| {
            (0..tangent.len()).for_each(|other| {
                tangent[index][other] = 0.0;
                tangent[other][index] = 0.0;
            });
            tangent[index][index] = 1.0;
        })
    }
}

pub(super) fn kkt_entry_sparse<H: Hessian, M: ConstraintMatrix>(
    hessian: &H,
    constraint_matrix: &M,
    num_variables: usize,
    row: usize,
    column: usize,
    eliminated: &Eliminated,
) -> Scalar {
    if row >= num_variables {
        if eliminated.contains(column) {
            0.0
        } else {
            -constraint_matrix.at(row - num_variables, column)
        }
    } else if column >= num_variables {
        if eliminated.contains(row) {
            0.0
        } else {
            -constraint_matrix.at(column - num_variables, row)
        }
    } else if eliminated.contains(row) || eliminated.contains(column) {
        if row == column { 1.0 } else { 0.0 }
    } else {
        hessian.entry(row, column)
    }
}
