use super::{Quantity, sparse_vec::QuantitySparseVec};
use crate::math::{
    Hessian, Scalar, SquareMatrix, Tensor, TensorRank0, Vector, assert::FiniteDifference,
    tensor::vec::TensorVector,
};
use crate::units::Dimensionless;

/// A vector of sparse vectors of quantities, storing only inserted entries.
///
/// What a stiffness is where a node carries one degree of freedom, the blocks a
/// rank-2 sparse vector stores being a single entry apiece.
pub type QuantitySparseVec2D<U = Dimensionless> = TensorVector<QuantitySparseVec<U>>;

impl<U> QuantitySparseVec2D<U> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| QuantitySparseVec::default()).collect()
    }
}

impl<U> Hessian for QuantitySparseVec2D<U> {
    fn quadratic_form(&self, vector: &Vector) -> Scalar {
        self.iter()
            .enumerate()
            .map(|(a, row)| {
                row.entries()
                    .map(|(b, entry)| entry.value() * vector[a] * vector[b])
                    .sum::<Scalar>()
            })
            .sum()
    }
    fn entry(&self, row: usize, column: usize) -> Scalar {
        match self[row].0.binary_search_by_key(&column, |&(b, _)| b) {
            Ok(k) => self[row].0[k].1.value(),
            Err(_) => 0.0,
        }
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries()
                .for_each(|(b, entry)| square_matrix[a][b] = entry.value())
        });
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        let mut remap = vec![0; retained.len()];
        let mut count = 0;
        retained.iter().enumerate().for_each(|(p, &keep)| {
            if keep {
                remap[p] = count;
                count += 1;
            }
        });
        let mut square_matrix = SquareMatrix::zero(count);
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, entry)| {
                if retained[a] && retained[b] {
                    square_matrix[remap[a]][remap[b]] = entry.value()
                }
            })
        });
        square_matrix
    }
}

impl<U> FiniteDifference for QuantitySparseVec2D<U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let zero = Quantity::new(0.0);
        let entry_errors = |self_ab: &Quantity<U>, comparator_ab: &Quantity<U>| {
            if self_ab.differs(*comparator_ab, epsilon) {
                (
                    1,
                    self_ab.differs_severely(*comparator_ab, epsilon) as usize,
                )
            } else {
                (0, 0)
            }
        };
        let (error_count, severe_count) = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_a, comparator_a)| {
                let mut errors = (0, 0);
                let (mut p, mut q) = (0, 0);
                while p < self_a.0.len() || q < comparator_a.0.len() {
                    let b = self_a.0.get(p).map(|&(b, _)| b);
                    let c = comparator_a.0.get(q).map(|&(c, _)| c);
                    let entry = match (b, c) {
                        (Some(b), Some(c)) if b == c => {
                            p += 1;
                            q += 1;
                            entry_errors(&self_a.0[p - 1].1, &comparator_a.0[q - 1].1)
                        }
                        (Some(b), Some(c)) if b < c => {
                            p += 1;
                            entry_errors(&self_a.0[p - 1].1, &zero)
                        }
                        (Some(_), None) => {
                            p += 1;
                            entry_errors(&self_a.0[p - 1].1, &zero)
                        }
                        _ => {
                            q += 1;
                            entry_errors(&zero, &comparator_a.0[q - 1].1)
                        }
                    };
                    errors.0 += entry.0;
                    errors.1 += entry.1;
                }
                errors
            })
            .fold((0, 0), |sum, errors| (sum.0 + errors.0, sum.1 + errors.1));
        if error_count > 0 {
            Some((severe_count > 0, error_count))
        } else {
            None
        }
    }
}
