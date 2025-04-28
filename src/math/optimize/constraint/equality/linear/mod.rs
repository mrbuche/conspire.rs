use crate::math::{SquareMatrix, Vector, TensorRank1Sparse, TensorVec};
use super::super::IntoConstraint;

/// ???
pub type LinearEqualityConstraint = (SquareMatrix, Vector);

impl<const D: usize, const I: usize> IntoConstraint<LinearEqualityConstraint> for TensorRank1Sparse<D, I> {
    fn into_constraint(self, num: usize) -> LinearEqualityConstraint {
        // let multipliers_len = self.len();
        // let full_length = 5;
        let mults = self.len();
        let mut kkt = SquareMatrix::zero(mults + D * num);
        // let mut matrix = Matrix::zero();
        self.iter().for_each(|(a, i, _)|
            kkt[mults + D * a + i][mults + D * a + i] = 1.0
            need to check this
        );
        let vector = self.iter().map(|(_, _, value)|
            *value
        ).collect();
        (kkt, vector)
    }
}

// trait FromSparse<T> {
//     fn from_sparse(sparse: T, num: usize) -> Self;
// }

// impl<const D: usize, const I: usize> FromSparse<TensorRank1Sparse<D, I>> for LinearEqualityConstraint {
//     fn from_sparse(sparse: TensorRank1Sparse<D, I>, num: usize) -> Self {
//         let matrix = sparse.iter().map(|(a, i, value)| {
//             let mut row = Vector::zero(num);
//             row[D * a + i] = 1.0;
//             row
//         }).collect();
//         let vector = sparse.iter().map(|(_, _, value)|
//             *value
//         ).collect();
//         Self {
//             matrix,
//             vector,
//         }
//     }
// }
