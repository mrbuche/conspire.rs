use crate::math::{SquareMatrix, Vector, TensorRank1Sparse, TensorVec};
use super::super::ToConstraint;

/// A linear equality constraint.
pub type LinearEqualityConstraint = (SquareMatrix, Vector);

impl<const D: usize, const I: usize> ToConstraint<LinearEqualityConstraint> for TensorRank1Sparse<D, I> {
    fn to_constraint(&self, num: usize) -> LinearEqualityConstraint {
        let num_dof = D * num;
        let num_constraints = self.iter().count();
        let mut kkt = SquareMatrix::zero(num_constraints + num_dof);
        self.iter().enumerate().for_each(|(index, (a, i, _))| {
            kkt[num_dof + index][D * a + i] = -1.0;
            kkt[D * a + i][num_dof + index] = -1.0
        });
        let rhs = self.iter().map(|(_, _, value)|
            *value
        ).collect();
        (kkt, rhs)
    }
}
