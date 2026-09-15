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
    /// Returns the matrix logarithm of the 3x3 tensor.
    ///
    /// Diagonal tensors go entrywise; symmetric tensors (exactly or up to
    /// round-off) through the spectral decomposition; anything within the
    /// series' radius through a truncated series; and a general tensor
    /// through inverse scaling and squaring of that series.
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
            return Ok(logm);
        }
        let norm = (self - &TensorRank2::identity()).norm();
        if norm < 1e-2 {
            return Ok(self.logm_series());
        }
        let transpose = self.transpose();
        if self.is_symmetric() || (self - &transpose).norm().value() < 1e-9 * (1.0 + norm.value()) {
            let symmetric = (self + transpose) * 0.5;
            let mut eigenvalues = solve_cubic_symmetric(symmetric.invariants())?;
            if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
                return Err(TensorError::NotPositiveDefinite);
            }
            let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, &symmetric);
            eigenvalues
                .iter_mut()
                .for_each(|eigenvalue| *eigenvalue = eigenvalue.ln());
            return Ok(reconstruct_symmetric(eigenvalues, eigenvectors));
        }
        //
        // Non-symmetric, outside the series' radius: inverse scaling and
        // squaring. Repeated matrix square roots (Denman-Beavers) bring
        // `self` within the series branch's radius; log(A) = 2^m log(A^{1/2^m}).
        //
        let mut root = self.clone();
        let mut squarings: i32 = 0;
        while (&root - &TensorRank2::identity()).norm().value() >= 1e-2 {
            root = root.sqrtm()?;
            squarings += 1;
        }
        Ok(root.logm_series() * 2.0_f64.powi(squarings))
    }
    /// The truncated series `-Σ (-(A-I))ᵏ/k`; accurate only within a small
    /// norm of the identity. Pure matrix products, no symmetry needed.
    fn logm_series(&self) -> Self {
        let tensor = self - &TensorRank2::identity();
        let norm = tensor.norm();
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
        logm
    }
    /// Returns a principal square root of the 3x3 tensor via the
    /// Denman-Beavers iteration `Y_{k+1} = (Y_k + Z_k⁻¹)/2, Z_{k+1} = (Z_k +
    /// Y_k⁻¹)/2` (`Y_0 = A, Z_0 = I`), which converges quadratically to
    /// `(√A, √A⁻¹)` for a matrix with no eigenvalues on the non-positive real
    /// axis.
    fn sqrtm(&self) -> Result<Self, TensorError> {
        let mut y = self.clone();
        let mut z = Self::identity();
        for _ in 0..64 {
            let y_inverse = y.inverse();
            let z_inverse = z.inverse();
            let y_next = (&y + z_inverse) * 0.5;
            let z_next = (&z + y_inverse) * 0.5;
            if (&y_next - &y).norm().value() < 1e-13 * (1.0 + y_next.norm().value()) {
                return Ok(y_next);
            }
            y = y_next;
            z = z_next;
        }
        Err(TensorError::SquareRootDidNotConverge)
    }
    /// Returns a principal square root of the 3x3 tensor together with its
    /// Fréchet derivative, by forward-differentiating the same
    /// Denman-Beavers iteration [`Self::sqrtm`] uses: with
    /// `d(X⁻¹)[H] = -X⁻¹HX⁻¹` carried as a rank-4 operator (via
    /// [`sandwich_negated`]), `dY_{k+1} = (dY_k + d(Z_k⁻¹))/2`,
    /// `dZ_{k+1} = (dZ_k + d(Y_k⁻¹))/2`, seeded `dY_0 = I⊗I`, `dZ_0 = 0`.
    #[allow(clippy::type_complexity)]
    fn dsqrtm(&self) -> Result<(Self, TensorRank4<3, I, I, I, I, Dimensionless>), TensorError> {
        let mut y = self.clone();
        let mut z = Self::identity();
        let mut dy: TensorRank4<3, I, I, I, I, Dimensionless> =
            TensorRank4::dyad_ik_jl(&Self::identity(), &Self::identity());
        let mut dz = TensorRank4::zero();
        for _ in 0..64 {
            let y_inverse = y.inverse();
            let z_inverse = z.inverse();
            let dy_inverse = sandwich_negated(&y_inverse, &dy);
            let dz_inverse = sandwich_negated(&z_inverse, &dz);
            let y_next = (&y + z_inverse) * 0.5;
            let z_next = (&z + y_inverse) * 0.5;
            let dy_next = (dy + dz_inverse) * 0.5;
            let dz_next = (dz + dy_inverse) * 0.5;
            if (&y_next - &y).norm().value() < 1e-13 * (1.0 + y_next.norm().value()) {
                return Ok((y_next, dy_next));
            }
            y = y_next;
            z = z_next;
            dy = dy_next;
            dz = dz_next;
        }
        Err(TensorError::SquareRootDidNotConverge)
    }
    /// The derivative of [`Self::logm_series`]; the same power-derivative
    /// identity `d(Aᵏ)[H] = Σₚ Aᵖ H Aᵏ⁻¹⁻ᵖ` used by `dexpm`'s small-norm
    /// branch, applied to `A - I` with the log series' coefficients.
    fn dlogm_series(&self) -> TensorRank4<3, I, I, I, I, Dimensionless> {
        let tensor = self - &TensorRank2::identity();
        let norm = tensor.norm();
        let num_terms = if norm < 1e-4 {
            2
        } else if norm < 1e-3 {
            3
        } else {
            5
        };
        let mut power = Self::identity();
        let mut powers = vec![power.clone()];
        (1..num_terms).for_each(|_| {
            power *= &tensor;
            powers.push(power.clone())
        });
        let mut dlogm = TensorRank4::zero();
        for n in 1..=num_terms {
            let coefficient = (if n % 2 == 0 { -1.0 } else { 1.0 }) / n as f64;
            for p in 0..n {
                let (left, right) = (&powers[p], &powers[n - 1 - p]);
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            for l in 0..3 {
                                dlogm[i][j][k][l] += Quantity::new(
                                    left[i][k].value() * right[l][j].value() * coefficient,
                                )
                            }
                        }
                    }
                }
            }
        }
        dlogm
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
            //
            // Non-symmetric: chain rule through the same inverse scaling and
            // squaring as logm, composing dsqrtm's rank-4 Jacobian at each
            // square root with the accumulated derivative so far.
            //
            let mut root = self.clone();
            let mut total_derivative: TensorRank4<3, I, I, I, I, Dimensionless> =
                TensorRank4::dyad_ik_jl(&Self::identity(), &Self::identity());
            let mut squarings: i32 = 0;
            while (&root - &TensorRank2::identity()).norm().value() >= 1e-2 {
                let (next_root, derivative) = root.dsqrtm()?;
                total_derivative = compose(&derivative, &total_derivative);
                root = next_root;
                squarings += 1;
            }
            Ok(compose(&root.dlogm_series(), &total_derivative) * 2.0_f64.powi(squarings))
        }
    }
}

/// `-middle · d · middle` with `d` a rank-4 operator: the Fréchet derivative
/// of `X ↦ X⁻¹` at `middle = X⁻¹`, `d(X⁻¹)[H] = -X⁻¹HX⁻¹`, applied to every
/// direction `d` carries at once.
fn sandwich_negated<I>(
    middle: &TensorRank2<3, I, I, Dimensionless>,
    d: &TensorRank4<3, I, I, I, I, Dimensionless>,
) -> TensorRank4<3, I, I, I, I, Dimensionless> {
    let mut result = TensorRank4::zero();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let mut value = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            value +=
                                middle[i][a].value() * d[a][b][k][l].value() * middle[b][j].value();
                        }
                    }
                    result[i][j][k][l] = Quantity::new(-value);
                }
            }
        }
    }
    result
}

/// Composes two rank-4 operators, contracting `outer`'s last two indices with
/// `inner`'s first two: `(outer ∘ inner)[i][j][k][l] = Σ outer[i][j][a][b]
/// inner[a][b][k][l]` — the chain rule for two linearizations in sequence.
fn compose<I>(
    outer: &TensorRank4<3, I, I, I, I, Dimensionless>,
    inner: &TensorRank4<3, I, I, I, I, Dimensionless>,
) -> TensorRank4<3, I, I, I, I, Dimensionless> {
    let mut result = TensorRank4::zero();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let mut value = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            value += outer[i][j][a][b].value() * inner[a][b][k][l].value();
                        }
                    }
                    result[i][j][k][l] = Quantity::new(value);
                }
            }
        }
    }
    result
}
