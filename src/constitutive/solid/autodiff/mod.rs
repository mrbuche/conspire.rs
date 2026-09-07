//! Shared plain-`f64` plumbing for autodiff solid constitutive kernels:
//! `DeformationGradient` <-> row-major `[f64; 9]`, 3x3 determinant / inverse,
//! and the stress-measure push-forwards. Used by both the `elastic::autodiff`
//! and `hyperelastic::autodiff` model kernels.

#![allow(clippy::needless_range_loop)]

use crate::{math::TensorArray, mechanics::DeformationGradient};

/// Row-major, `[3 * i + j] = F_iJ`.
pub(crate) fn flatten(deformation_gradient: &DeformationGradient) -> [f64; 9] {
    let a = deformation_gradient.as_array();
    [
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ]
}

/// `det F`, `f` row-major.
pub(crate) fn determinant(f: &[f64; 9]) -> f64 {
    f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6])
}

/// `F^-1`, row-major.
pub(crate) fn inverse(f: &[f64; 9]) -> [f64; 9] {
    let jacobian = determinant(f);
    [
        (f[4] * f[8] - f[5] * f[7]) / jacobian,
        (f[2] * f[7] - f[1] * f[8]) / jacobian,
        (f[1] * f[5] - f[2] * f[4]) / jacobian,
        (f[5] * f[6] - f[3] * f[8]) / jacobian,
        (f[0] * f[8] - f[2] * f[6]) / jacobian,
        (f[2] * f[3] - f[0] * f[5]) / jacobian,
        (f[3] * f[7] - f[4] * f[6]) / jacobian,
        (f[1] * f[6] - f[0] * f[7]) / jacobian,
        (f[0] * f[4] - f[1] * f[3]) / jacobian,
    ]
}

/// `sigma = J^-1 P F^T` from first Piola-Kirchhoff `p`, all row-major.
pub(crate) fn push_cauchy(p: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let jacobian = determinant(f);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] =
                (p[3 * i] * f[3 * j] + p[3 * i + 1] * f[3 * j + 1] + p[3 * i + 2] * f[3 * j + 2])
                    / jacobian;
        }
    }
}

/// `S = F^-1 P` from first Piola-Kirchhoff `p`, all row-major.
pub(crate) fn push_second_piola(p: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let f_inverse = inverse(f);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = f_inverse[3 * i] * p[j]
                + f_inverse[3 * i + 1] * p[3 + j]
                + f_inverse[3 * i + 2] * p[6 + j];
        }
    }
}

/// `P = J sigma F^-T` from Cauchy stress `sigma`, all row-major.
pub(crate) fn push_first_piola(sigma: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let (jacobian, f_inverse) = (determinant(f), inverse(f));
    for i in 0..3 {
        for k in 0..3 {
            out[3 * i + k] = jacobian
                * (sigma[3 * i] * f_inverse[3 * k]
                    + sigma[3 * i + 1] * f_inverse[3 * k + 1]
                    + sigma[3 * i + 2] * f_inverse[3 * k + 2]);
        }
    }
}
