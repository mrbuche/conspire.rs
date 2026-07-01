#[cfg(test)]
mod test;

use std::f64::consts::TAU;

use super::{
    super::{
        super::assert_eq_within_tols,
        Rank2, Tensor, TensorArray, TensorError,
        rank_0::{TensorRank0, list::TensorRank0List},
        rank_1::{CrossProduct, TensorRank1},
        rank_4::TensorRank4,
    },
    TensorRank2,
};
use crate::ABS_TOL;

impl<const I: usize> TensorRank2<3, I, I> {
    /// Returns the matrix logarithm of the 3x3 symmetric tensor.
    pub fn logm(&self) -> Result<Self, TensorError> {
        if self.is_diagonal() {
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
                    panic!("Symmetric matrix has a non-positive eigenvalue")
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
    pub fn dlogm(&self) -> Result<TensorRank4<3, I, I, I, I>, TensorError> {
        if self.is_diagonal() {
            let mut dlogm = TensorRank4::zero();
            dlogm.iter_mut().enumerate().for_each(|(i, dlogm_i)| {
                dlogm_i.iter_mut().enumerate().for_each(|(j, dlogm_ij)| {
                    dlogm_ij.iter_mut().enumerate().for_each(|(k, dlogm_ijk)| {
                        dlogm_ijk
                            .iter_mut()
                            .enumerate()
                            .filter(|(l, _)| i == k && &j == l)
                            .for_each(|(_, dlogm_ijkl)| {
                                *dlogm_ijkl = if assert_eq_within_tols(&self[i][i], &self[j][j])
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
                panic!("Symmetric matrix has a non-positive eigenvalue")
            }
            let divided_difference: Self = eigenvalues
                .iter()
                .map(|eigenvalue_i| {
                    eigenvalues
                        .iter()
                        .map(|eigenvalue_j| {
                            if assert_eq_within_tols(eigenvalue_i, eigenvalue_j).is_ok() {
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
                                ).sum::<TensorRank0>()
                            ).sum()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect())
        } else {
            panic!("Matrix logarithm only implemented for symmetric cases")
        }
    }
    /// Returns the invariants of the 3x3 symmetric tensor.
    pub fn invariants(&self) -> TensorRank0List<3> {
        let trace = self.trace();
        TensorRank0List::from([
            trace,
            0.5 * (trace.powi(2) - self.squared_trace()),
            self.determinant(),
        ])
    }
}

fn solve_cubic_symmetric(
    coefficients: TensorRank0List<3>,
) -> Result<TensorRank0List<3>, TensorError> {
    let c2 = coefficients[0];
    let c1 = coefficients[1];
    let c0 = coefficients[2];
    let p = c1 - c2 * c2 / 3.0;
    let q = -(2.0 * c2.powi(3) - 9.0 * c2 * c1 + 27.0 * c0) / 27.0;
    if p.abs() < ABS_TOL {
        let t = (-q).cbrt();
        let lambda = t + c2 / 3.0;
        return Ok(TensorRank0List::from([lambda; _]));
    }
    let discriminant = -4.0 * p * p * p - 27.0 * q * q;
    if discriminant >= ABS_TOL {
        let sqrt_term = (-p / 3.0).sqrt();
        let cos_arg = 3.0 * q / (2.0 * p * (-p / 3.0).sqrt());
        let cos_arg = cos_arg.clamp(-1.0, 1.0);
        let theta = cos_arg.acos();
        let mut lambdas = [
            2.0 * sqrt_term * (theta / 3.0).cos() + c2 / 3.0,
            2.0 * sqrt_term * ((theta + TAU) / 3.0).cos() + c2 / 3.0,
            2.0 * sqrt_term * ((theta + 2.0 * TAU) / 3.0).cos() + c2 / 3.0,
        ];
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Ok(TensorRank0List::from(lambdas))
    } else {
        Err(TensorError::SymmetricMatrixComplexEigenvalues)
    }
}

fn find_orthonormal_eigenvectors<const I: usize>(
    eigenvalues: &TensorRank0List<3>,
    tensor: &TensorRank2<3, I, I>,
) -> TensorRank2<3, I, I> {
    let mut eigenvectors = eigenvalues
        .iter()
        .map(|&eigenvalue| eigenvector_symmetric(eigenvalue, tensor))
        .collect::<TensorRank2<3, I, I>>();
    eigenvectors[0].normalize();
    let proj1 = &eigenvectors[1] * &eigenvectors[0];
    for i in 0..3 {
        eigenvectors[1][i] -= proj1 * eigenvectors[0][i];
    }
    eigenvectors[1].normalize();
    eigenvectors[2] = eigenvectors[0].cross(&eigenvectors[1]);
    eigenvectors
}

fn eigenvector_symmetric<const I: usize>(
    eigenvalue: TensorRank0,
    tensor: &TensorRank2<3, I, I>,
) -> TensorRank1<3, I> {
    let m = tensor - TensorRank2::identity() * eigenvalue;
    let mut pivot_row = 0;
    m.iter().enumerate().for_each(|(i, m_i)| {
        if m_i[i].abs() < m[pivot_row][pivot_row].abs() {
            pivot_row = i;
        }
    });
    if pivot_row == 0 {
        m[1].cross(&m[2])
    } else if pivot_row == 1 {
        m[0].cross(&m[2])
    } else {
        m[0].cross(&m[1])
    }
    .normalized()
}

fn reconstruct_symmetric<const I: usize>(
    eigenvalues: TensorRank0List<3>,
    eigenvectors: TensorRank2<3, I, I>,
) -> TensorRank2<3, I, I> {
    let mut tensor = TensorRank2::zero();
    eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .for_each(|(eigenvalue, eigenvector)| {
            tensor
                .iter_mut()
                .zip(eigenvector.iter())
                .for_each(|(tensor_i, eigenvector_i)| {
                    tensor_i.iter_mut().zip(eigenvector.iter()).for_each(
                        |(tensor_ij, eigenvector_j)| {
                            *tensor_ij += eigenvalue * eigenvector_i * eigenvector_j
                        },
                    )
                })
        });
    tensor
}
