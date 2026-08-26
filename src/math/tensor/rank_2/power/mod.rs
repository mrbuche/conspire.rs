use crate::math::Quantity;
use crate::units::Dimensionless;

use super::{
    super::{
        Rank2, Tensor, TensorArray, TensorError,
        rank_0::{TensorRank0, list::TensorRank0List},
        rank_4::TensorRank4,
    },
    TensorRank2,
    eigen::reconstruct_symmetric,
};
use crate::math::assert::Assert;

impl<I> TensorRank2<3, I, I, Dimensionless> {
    /// Returns the matrix power of the 3x3 symmetric tensor.
    pub fn powm(&self, exponent: TensorRank0) -> Result<Self, TensorError> {
        if self.is_diagonal() {
            let mut powm = TensorRank2::zero();
            powm.iter_mut()
                .enumerate()
                .zip(self.iter())
                .for_each(|((i, powm_i), self_i)| powm_i[i] = self_i[i].powf(exponent));
            Ok(powm)
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
                let mut powm = TensorRank2::identity();
                let mut term = tensor.clone();
                let mut coefficient = exponent;
                (1..=num_terms).for_each(|k| {
                    powm += &term * coefficient;
                    term *= &tensor;
                    coefficient *= (exponent - k as TensorRank0) / (k as TensorRank0 + 1.0);
                });
                Ok(powm)
            } else if self.is_symmetric() {
                let (eigenvalues, eigenvectors) = self.eigen()?;
                Ok(Self::powm_from_eigen(&eigenvalues, &eigenvectors, exponent))
            } else {
                panic!("Matrix power only implemented for symmetric cases")
            }
        }
    }
    /// Returns the matrix power from an eigendecomposition obtained from [`Self::eigen`].
    pub fn powm_from_eigen(
        eigenvalues: &TensorRank0List<3>,
        eigenvectors: &Self,
        exponent: TensorRank0,
    ) -> Self {
        if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
            panic!("Symmetric matrix has a non-positive eigenvalue")
        }
        let powered = eigenvalues
            .iter()
            .map(|eigenvalue| eigenvalue.powf(exponent))
            .collect();
        reconstruct_symmetric(powered, eigenvectors.clone())
    }
    /// Returns the derivative of the matrix power of the 3x3 symmetric tensor.
    pub fn dpowm(
        &self,
        exponent: TensorRank0,
    ) -> Result<TensorRank4<3, I, I, I, I, Dimensionless>, TensorError> {
        if self.is_diagonal() {
            let mut dpowm = TensorRank4::zero();
            dpowm.iter_mut().enumerate().for_each(|(i, dpowm_i)| {
                dpowm_i.iter_mut().enumerate().for_each(|(j, dpowm_ij)| {
                    dpowm_ij.iter_mut().enumerate().for_each(|(k, dpowm_ijk)| {
                        dpowm_ijk
                            .iter_mut()
                            .enumerate()
                            .filter(|(l, _)| i == k && &j == l)
                            .for_each(|(_, dpowm_ijkl)| {
                                *dpowm_ijkl = if Assert::default()
                                    .eq_within_tols(self[i][i], &self[j][j])
                                    .is_ok()
                                {
                                    exponent * self[j][j].powf(exponent - 1.0)
                                } else {
                                    (self[i][i].powf(exponent) - self[j][j].powf(exponent))
                                        / (self[i][i] - self[j][j])
                                }
                            })
                    })
                })
            });
            Ok(dpowm)
        } else if self.is_symmetric() {
            let (eigenvalues, eigenvectors) = self.eigen()?;
            Ok(Self::dpowm_from_eigen(
                &eigenvalues,
                &eigenvectors,
                exponent,
            ))
        } else {
            panic!("Matrix power only implemented for symmetric cases")
        }
    }
    /// Returns the derivative of the matrix power from an eigendecomposition obtained from
    /// [`Self::eigen`].
    pub fn dpowm_from_eigen(
        eigenvalues: &TensorRank0List<3>,
        eigenvectors: &Self,
        exponent: TensorRank0,
    ) -> TensorRank4<3, I, I, I, I, Dimensionless> {
        if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
            panic!("Symmetric matrix has a non-positive eigenvalue")
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
                            exponent * eigenvalue_j.powf(exponent - 1.0)
                        } else {
                            (eigenvalue_i.powf(exponent) - eigenvalue_j.powf(exponent))
                                / (eigenvalue_i - eigenvalue_j)
                        }
                    })
                    .collect()
            })
            .collect();
        let eigenvectors_transposed = eigenvectors.transpose();
        eigenvectors_transposed.iter().map(|eigenvector_i|
            eigenvectors_transposed.iter().map(|eigenvector_j|
                eigenvectors_transposed.iter().map(|eigenvector_k|
                    eigenvectors_transposed.iter().map(|eigenvector_l|
                        eigenvector_i.iter().zip(eigenvector_k.iter().zip(divided_difference.iter())).map(|(eigenvector_ip, (eigenvector_kp, divided_difference_p))|
                            eigenvector_j.iter().zip(eigenvector_l.iter().zip(divided_difference_p.iter())).map(|(eigenvector_jq, (eigenvector_lq, divided_difference_pq))|
                                eigenvector_ip * eigenvector_kp * divided_difference_pq * eigenvector_jq * eigenvector_lq
                            ).sum::<Quantity>()
                        ).sum::<Quantity>()
                    ).collect()
                ).collect()
            ).collect()
        ).collect()
    }
}

/// A cached eigendecomposition of a symmetric tensor, letting [`Self::powm`]/[`Self::dpowm`] be
/// evaluated at several exponents while paying for only one cubic eigensolve.
pub enum Spectrum<I> {
    Eigen(TensorRank0List<3>, TensorRank2<3, I, I, Dimensionless>),
    Fallback(TensorRank2<3, I, I, Dimensionless>),
}

impl<I> Spectrum<I> {
    /// Caches the eigendecomposition of the tensor, when one is needed.
    pub fn new(tensor: &TensorRank2<3, I, I, Dimensionless>) -> Result<Self, TensorError> {
        if tensor.is_diagonal() || (tensor - &TensorRank2::identity()).norm() < 1e-2 {
            Ok(Self::Fallback(tensor.clone()))
        } else {
            let (eigenvalues, eigenvectors) = tensor.eigen()?;
            Ok(Self::Eigen(eigenvalues, eigenvectors))
        }
    }
    /// Returns the matrix power at the given exponent, reusing the cached eigendecomposition.
    pub fn powm(
        &self,
        exponent: TensorRank0,
    ) -> Result<TensorRank2<3, I, I, Dimensionless>, TensorError> {
        Ok(match self {
            Self::Eigen(eigenvalues, eigenvectors) => {
                TensorRank2::powm_from_eigen(eigenvalues, eigenvectors, exponent)
            }
            Self::Fallback(tensor) => tensor.powm(exponent)?,
        })
    }
    /// Returns the derivative of the matrix power at the given exponent, reusing the cached
    /// eigendecomposition.
    pub fn dpowm(
        &self,
        exponent: TensorRank0,
    ) -> Result<TensorRank4<3, I, I, I, I, Dimensionless>, TensorError> {
        Ok(match self {
            Self::Eigen(eigenvalues, eigenvectors) => {
                TensorRank2::dpowm_from_eigen(eigenvalues, eigenvectors, exponent)
            }
            Self::Fallback(tensor) => tensor.dpowm(exponent)?,
        })
    }
}
