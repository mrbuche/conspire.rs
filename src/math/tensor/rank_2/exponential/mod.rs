#[cfg(test)]
mod test;

use crate::math::Quantity;
use crate::math::assert::Assert;
use crate::units::Dimensionless;

use super::{
    super::{Rank2, Tensor, TensorArray, TensorError, rank_4::TensorRank4},
    TensorRank2,
    eigen::{find_orthonormal_eigenvectors, reconstruct_symmetric, solve_cubic_symmetric},
};

impl<I> TensorRank2<3, I, I, Dimensionless> {
    /// Returns the matrix exponential of the 3x3 tensor.
    ///
    /// Diagonal tensors go entrywise; symmetric tensors (exactly or up to
    /// round-off) through the spectral decomposition; anything with a small
    /// enough norm through a truncated Taylor series; and a general tensor
    /// through scaling and squaring of that series.
    pub fn expm(&self) -> Result<Self, TensorError> {
        if self.is_diagonal() {
            let mut expm = TensorRank2::zero();
            expm.iter_mut()
                .enumerate()
                .zip(self.iter())
                .for_each(|((i, expm_i), self_i)| expm_i[i] = self_i[i].exp());
            return Ok(expm);
        }
        let norm = self.norm().value();
        if norm < 1e-2 {
            return Ok(self.expm_series());
        }
        let transpose = self.transpose();
        if self.is_symmetric() || (self - &transpose).norm().value() < 1e-9 * (1.0 + norm) {
            let symmetric = (self + transpose) * 0.5;
            let mut eigenvalues = solve_cubic_symmetric(symmetric.invariants())?;
            let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, &symmetric);
            eigenvalues
                .iter_mut()
                .for_each(|eigenvalue| *eigenvalue = eigenvalue.exp());
            return Ok(reconstruct_symmetric(eigenvalues, eigenvectors));
        }
        let squarings = (norm / 5e-3).log2().ceil().max(1.0) as u32;
        let mut expm = (self / 2.0_f64.powi(squarings as i32)).expm_series();
        (0..squarings).for_each(|_| expm = &expm * &expm);
        Ok(expm)
    }
    /// The truncated Taylor series `Σ Aᵏ/k!`; accurate only for a small norm.
    fn expm_series(&self) -> Self {
        let num_terms = match self.norm().value() {
            norm if norm < 1e-4 => 3,
            norm if norm < 1e-3 => 5,
            _ => 8,
        };
        let mut expm = self + TensorRank2::identity();
        let mut power = self.clone();
        let mut factorial = 1.0;
        (2..=num_terms).for_each(|k| {
            power *= self;
            factorial *= k as f64;
            expm += &power / factorial;
        });
        expm
    }
    /// Returns the derivative of the matrix exponential of the 3x3 tensor.
    ///
    /// The Frechet derivative $`\mathrm{d}\exp(\mathbf{A})/\mathrm{d}\mathbf{A}`$, formed
    /// diagonally entrywise, from a truncated series near zero, from scaling and
    /// squaring of that series for a general (non-symmetric, larger-norm) tensor,
    /// and otherwise from the spectral decomposition with the divided differences
    /// ```math
    /// \frac{e^{\lambda_i} - e^{\lambda_j}}{\lambda_i - \lambda_j},
    /// \qquad e^{\lambda_j} \text{ for } \lambda_i = \lambda_j.
    /// ```
    pub fn dexpm(&self) -> Result<TensorRank4<3, I, I, I, I, Dimensionless>, TensorError> {
        if self.is_diagonal() {
            let mut dexpm = TensorRank4::zero();
            dexpm.iter_mut().enumerate().for_each(|(i, dexpm_i)| {
                dexpm_i.iter_mut().enumerate().for_each(|(j, dexpm_ij)| {
                    dexpm_ij.iter_mut().enumerate().for_each(|(k, dexpm_ijk)| {
                        dexpm_ijk
                            .iter_mut()
                            .enumerate()
                            .filter(|(l, _)| i == k && &j == l)
                            .for_each(|(_, dexpm_ijkl)| {
                                *dexpm_ijkl = if Assert::default()
                                    .eq_within_tols(self[i][i], &self[j][j])
                                    .is_ok()
                                {
                                    self[j][j].exp()
                                } else {
                                    (self[i][i].exp() - self[j][j].exp())
                                        / (self[i][i] - self[j][j])
                                }
                            })
                    })
                })
            });
            Ok(dexpm)
        } else {
            let norm = self.norm().value();
            if norm < 1e-2 {
                //
                // d(A^n)[H] = sum_{p=0}^{n-1} A^p . H . A^{n-1-p}, so the truncated series
                // gives dexpm_{ijkl} = sum_n (1/n!) sum_p (A^p)_{ik} (A^{n-1-p})_{lj}.
                //
                let num_terms = if norm < 1e-4 {
                    3
                } else if norm < 1e-3 {
                    5
                } else {
                    8
                };
                let mut power = Self::identity();
                let mut powers = vec![power.clone()];
                (1..num_terms).for_each(|_| {
                    power *= self;
                    powers.push(power.clone())
                });
                let mut dexpm = TensorRank4::zero();
                let mut factorial = 1.0;
                for n in 1..=num_terms {
                    factorial *= n as f64;
                    for p in 0..n {
                        let (left, right) = (&powers[p], &powers[n - 1 - p]);
                        for i in 0..3 {
                            for j in 0..3 {
                                for k in 0..3 {
                                    for l in 0..3 {
                                        dexpm[i][j][k][l] += Quantity::new(
                                            left[i][k].value() * right[l][j].value() / factorial,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(dexpm)
            } else {
                let transpose = self.transpose();
                if !self.is_symmetric() && (self - &transpose).norm().value() >= 1e-9 * (1.0 + norm)
                {
                    //
                    // Non-symmetric: scaling and squaring of the Fréchet derivative.
                    // With E = exp(B), L = dexp(B), the squaring B → 2B gives
                    // E → E² and L → L·E + E·L (contracting the middle index);
                    // one final 1/scale converts d/dB back to d/dA.
                    //
                    let squarings = (norm / 5e-3).log2().ceil().max(1.0) as u32;
                    let scale = 2.0_f64.powi(squarings as i32);
                    let mut expm = (self / scale).expm_series();
                    let mut dexpm = (self / scale).dexpm()?;
                    for _ in 0..squarings {
                        let mut next = TensorRank4::zero();
                        for i in 0..3 {
                            for j in 0..3 {
                                for k in 0..3 {
                                    for l in 0..3 {
                                        let mut value = 0.0;
                                        for p in 0..3 {
                                            value += dexpm[i][p][k][l].value() * expm[p][j].value()
                                                + expm[i][p].value() * dexpm[p][j][k][l].value();
                                        }
                                        next[i][j][k][l] = Quantity::new(value);
                                    }
                                }
                            }
                        }
                        dexpm = next;
                        expm = &expm * &expm;
                    }
                    dexpm.iter_mut().for_each(|dexpm_i| {
                        dexpm_i.iter_mut().for_each(|dexpm_ij| {
                            dexpm_ij.iter_mut().for_each(|dexpm_ijk| {
                                dexpm_ijk
                                    .iter_mut()
                                    .for_each(|dexpm_ijkl| *dexpm_ijkl /= scale)
                            })
                        })
                    });
                    return Ok(dexpm);
                }
                let symmetric = (self + transpose) * 0.5;
                let eigenvalues = solve_cubic_symmetric(symmetric.invariants())?;
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
                                    eigenvalue_j.exp()
                                } else {
                                    (eigenvalue_i.exp() - eigenvalue_j.exp())
                                        / (eigenvalue_i - eigenvalue_j)
                                }
                            })
                            .collect()
                    })
                    .collect();
                let eigenvectors =
                    find_orthonormal_eigenvectors(&eigenvalues, &symmetric).transpose();
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
            }
        }
    }
    /// Applies the inverse matrix-exponential Fréchet derivative at `self` (the
    /// algebra element `σ`) to `rate`.
    ///
    /// The Bernoulli commutator series, truncated at four terms (exact to fifth
    /// order); with `\mathrm{ad}_\sigma(A) = \sigma A - A\sigma`,
    /// ```math
    /// \mathrm{dexpinv}_\sigma(A) = A - \tfrac{1}{2}[\sigma, A]
    ///     + \tfrac{1}{12}[\sigma, [\sigma, A]]
    ///     - \tfrac{1}{720}[\sigma, [\sigma, [\sigma, [\sigma, A]]]] .
    /// ```
    /// Pure matrix products — total on any input, no symmetry needed.
    pub fn dexpinv(&self, rate: &Self) -> Self {
        let mut term = rate.clone();
        let mut result = term.clone() * DEXPINV_COEFFICIENTS[0];
        for &coefficient in DEXPINV_COEFFICIENTS.iter().skip(1) {
            term = self * &term - &term * self;
            if coefficient != 0.0 {
                result += term.clone() * coefficient;
            }
        }
        result
    }
    /// The directional derivative of [`Self::dexpinv`] in the direction
    /// `(d_sigma, d_rate)`.
    ///
    /// Forward-mode through the same truncated series: with `T_0 = A` and
    /// `T_k = [\sigma, T_{k-1}]`,
    /// ```math
    /// \mathrm{d}T_0 = \mathrm{d}A, \qquad
    /// \mathrm{d}T_k = [\mathrm{d}\sigma, T_{k-1}] + [\sigma, \mathrm{d}T_{k-1}] .
    /// ```
    pub fn dexpinv_tangent(&self, rate: &Self, d_sigma: &Self, d_rate: &Self) -> Self {
        let mut term = rate.clone();
        let mut d_term = d_rate.clone();
        let mut result = d_term.clone() * DEXPINV_COEFFICIENTS[0];
        for &coefficient in DEXPINV_COEFFICIENTS.iter().skip(1) {
            d_term = d_sigma * &term - &term * d_sigma + (self * &d_term - &d_term * self);
            term = self * &term - &term * self;
            if coefficient != 0.0 {
                result += d_term.clone() * coefficient;
            }
        }
        result
    }
}

/// The Bernoulli numbers `Bₖ/k!` of the `dexpinv` series, truncated at four terms.
const DEXPINV_COEFFICIENTS: [f64; 5] = [1.0, -0.5, 1.0 / 12.0, 0.0, -1.0 / 720.0];
