#[cfg(test)]
mod test;

use crate::math::Quantity;
use crate::units::Dimensionless;

use super::{
    super::{Rank2, Tensor, TensorArray, TensorError, rank_4::TensorRank4},
    TensorRank2,
    eigen::{find_orthonormal_eigenvectors, reconstruct_symmetric, solve_cubic_symmetric},
};
use crate::math::assert::Assert;

impl<I> TensorRank2<3, I, I, Dimensionless> {
    /// Returns the matrix logarithm of the 3x3 symmetric tensor.
    pub fn logm(&self) -> Result<Self, TensorError> {
        if self.is_diagonal() {
            if self.iter().enumerate().any(|(i, self_i)| self_i[i] <= 0.0) {
                return Err(TensorError::NotPositiveDefinite);
            }
            let mut logm = TensorRank2::zero();
            logm.iter_mut()
                .enumerate()
                .zip(self.iter())
                .for_each(|((i, logm_i), self_i)| logm_i[i] = self_i[i].ln());
            Ok(logm)
        } else {
            let tensor = self - &TensorRank2::identity();
            let norm = tensor.norm();
            if norm < 1e-2 {
                let num_terms = if norm < 1e-4 {
                    2
                } else if norm < 1e-3 {
                    3
                } else {
                    5
                };
                let mut logm = tensor.clone();
                let mut power = tensor.clone();
                (2..=num_terms).for_each(|k| {
                    power *= &tensor;
                    logm += &power * (if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64);
                });
                Ok(logm)
            } else if self.is_symmetric() {
                let mut eigenvalues = solve_cubic_symmetric(self.invariants())?;
                if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
                    return Err(TensorError::NotPositiveDefinite);
                }
                let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, self);
                eigenvalues
                    .iter_mut()
                    .for_each(|eigenvalue| *eigenvalue = eigenvalue.ln());
                Ok(reconstruct_symmetric(eigenvalues, eigenvectors))
            } else {
                panic!("Matrix logarithm only implemented for symmetric cases")
            }
        }
    }
    /// Returns the derivative of the matrix logarithm of the 3x3 symmetric tensor.
    pub fn dlogm(&self) -> Result<TensorRank4<3, I, I, I, I, Dimensionless>, TensorError> {
        if self.is_diagonal() {
            if self.iter().enumerate().any(|(i, self_i)| self_i[i] <= 0.0) {
                return Err(TensorError::NotPositiveDefinite);
            }
            let mut dlogm = TensorRank4::zero();
            dlogm.iter_mut().enumerate().for_each(|(i, dlogm_i)| {
                dlogm_i.iter_mut().enumerate().for_each(|(j, dlogm_ij)| {
                    dlogm_ij.iter_mut().enumerate().for_each(|(k, dlogm_ijk)| {
                        dlogm_ijk
                            .iter_mut()
                            .enumerate()
                            .filter(|(l, _)| i == k && &j == l)
                            .for_each(|(_, dlogm_ijkl)| {
                                *dlogm_ijkl = if Assert::default()
                                    .eq_within_tols(self[i][i], &self[j][j])
                                    .is_ok()
                                {
                                    1.0 / self[j][j]
                                } else {
                                    (self[i][i].ln() - self[j][j].ln()) / (self[i][i] - self[j][j])
                                }
                            })
                    })
                })
            });
            Ok(dlogm)
        } else if self.is_symmetric() {
            let eigenvalues = solve_cubic_symmetric(self.invariants())?;
            if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
                return Err(TensorError::NotPositiveDefinite);
            }
            let divided_difference: Self = eigenvalues
                .iter()
                .map(|eigenvalue_i| {
                    eigenvalues
                        .iter()
                        .map(|eigenvalue_j| {
                            if Assert::default()
                                .eq_within_tols(eigenvalue_i, eigenvalue_j)
                                .is_ok()
                            {
                                1.0 / eigenvalue_j
                            } else {
                                (eigenvalue_i.ln() - eigenvalue_j.ln())
                                    / (eigenvalue_i - eigenvalue_j)
                            }
                        })
                        .collect()
                })
                .collect();
            let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, self).transpose();
            Ok(eigenvectors.iter().map(|eigenvector_i|
                eigenvectors.iter().map(|eigenvector_j|
                    eigenvectors.iter().map(|eigenvector_k|
                        eigenvectors.iter().map(|eigenvector_l|
                            eigenvector_i.iter().zip(eigenvector_k.iter().zip(divided_difference.iter())).map(|(eigenvector_ip, (eigenvector_kp, divided_difference_p))|
                                eigenvector_j.iter().zip(eigenvector_l.iter().zip(divided_difference_p.iter())).map(|(eigenvector_jq, (eigenvector_lq, divided_difference_pq))|
                                    eigenvector_ip * eigenvector_kp * divided_difference_pq * eigenvector_jq * eigenvector_lq
                                ).sum::<Quantity>()
                            ).sum::<Quantity>()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect())
        } else {
            panic!("Matrix logarithm only implemented for symmetric cases")
        }
    }
}
