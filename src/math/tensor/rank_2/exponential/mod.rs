#[cfg(test)]
mod test;

use crate::units::Dimensionless;

use super::{
    super::{Rank2, Tensor, TensorArray, TensorError},
    TensorRank2,
    eigen::{find_orthonormal_eigenvectors, reconstruct_symmetric, solve_cubic_symmetric},
};

impl<I> TensorRank2<3, I, I, Dimensionless> {
    /// Returns the matrix exponential of the 3x3 tensor.
    ///
    /// Implemented for diagonal and symmetric tensors: diagonal entrywise, and
    /// symmetric through the spectral decomposition, with a truncated Taylor
    /// series near zero. A tensor that is symmetric only up to round-off is
    /// symmetrized; a materially non-symmetric tensor panics.
    pub fn expm(&self) -> Result<Self, TensorError> {
        if self.is_diagonal() {
            let mut expm = TensorRank2::zero();
            expm.iter_mut()
                .enumerate()
                .zip(self.iter())
                .for_each(|((i, expm_i), self_i)| expm_i[i] = self_i[i].exp());
            Ok(expm)
        } else {
            let norm = self.norm().value();
            if norm < 1e-2 {
                let num_terms = if norm < 1e-4 {
                    3
                } else if norm < 1e-3 {
                    5
                } else {
                    8
                };
                let mut expm = self + TensorRank2::identity();
                let mut power = self.clone();
                let mut factorial = 1.0;
                (2..=num_terms).for_each(|k| {
                    power *= self;
                    factorial *= k as f64;
                    expm += &power / factorial;
                });
                Ok(expm)
            } else {
                let transpose = self.transpose();
                if !self.is_symmetric() && (self - &transpose).norm().value() >= 1e-9 * (1.0 + norm)
                {
                    panic!("Matrix exponential only implemented for symmetric cases")
                }
                let symmetric = (self + transpose) * 0.5;
                let mut eigenvalues = solve_cubic_symmetric(symmetric.invariants())?;
                let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, &symmetric);
                eigenvalues
                    .iter_mut()
                    .for_each(|eigenvalue| *eigenvalue = eigenvalue.exp());
                Ok(reconstruct_symmetric(eigenvalues, eigenvectors))
            }
        }
    }
}
