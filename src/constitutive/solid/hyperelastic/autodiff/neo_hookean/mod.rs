#[cfg(test)]
mod test;

use super::{AutodiffElastic, AutodiffHyperelastic};
use crate::{math::Quantity, units::Stress};
use std::autodiff::{autodiff_forward, autodiff_reverse};

/// [`NeoHookean`](crate::constitutive::solid::hyperelastic::NeoHookean) as
/// autodiff kernels; wrap in [`Autodiff`](super::Autodiff) for the `Elastic` /
/// `Hyperelastic` API.
#[derive(Clone, Debug)]
pub struct AutodiffNeoHookean {
    /// The bulk modulus.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus.
    pub shear_modulus: Quantity<Stress>,
}

impl AutodiffElastic for AutodiffNeoHookean {
    type Parameters = [f64; 2];
    fn parameters(&self) -> [f64; 2] {
        [self.bulk_modulus.value(), self.shear_modulus.value()]
    }
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
    fn cauchy(p: &[f64], f: &[f64; 9], out: &mut [f64; 9]) {
        cauchy(p[0], p[1], f, out)
    }
    fn piola(p: &[f64], f: &[f64; 9], out: &mut [f64; 9]) {
        piola(p[0], p[1], f, out)
    }
    fn second_piola(p: &[f64], f: &[f64; 9], out: &mut [f64; 9]) {
        second_piola(p[0], p[1], f, out)
    }
    fn cauchy_tangent(
        p: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_cauchy(p[0], p[1], f, df, primal, seed)
    }
    fn piola_tangent(
        p: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_piola(p[0], p[1], f, df, primal, seed)
    }
    fn second_piola_tangent(
        p: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_second_piola(p[0], p[1], f, df, primal, seed)
    }
}

impl AutodiffHyperelastic for AutodiffNeoHookean {
    fn energy(p: &[f64], f: &[f64; 9]) -> f64 {
        energy(p[0], p[1], f)
    }
}

fn determinant(f: &[f64; 9]) -> f64 {
    f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6])
}

/// Helmholtz free energy density, `f` row-major.
///
/// Mirrors `<NeoHookean as Hyperelastic>::helmholtz_free_energy_density`.
#[autodiff_reverse(d_energy, Const, Const, Duplicated, Active)]
fn energy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9]) -> f64 {
    let mut trace_b = 0.0;
    for k in 0..9 {
        trace_b += f[k] * f[k];
    }
    let jacobian = determinant(f);
    0.5 * (shear_modulus * (trace_b * jacobian.powf(-2.0 / 3.0) - 3.0)
        + bulk_modulus * (0.5 * (jacobian * jacobian - 1.0) - jacobian.ln()))
}

/// `P_iJ = dPsi/dF_iJ`, reverse mode.
#[autodiff_forward(d_piola, Const, Const, Dual, Dual)]
fn piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    for out_i in out.iter_mut() {
        *out_i = 0.0;
    }
    d_energy(bulk_modulus, shear_modulus, f, out, 1.0);
}

/// `sigma = J^-1 P F^T`.
#[autodiff_forward(d_cauchy, Const, Const, Dual, Dual)]
fn cauchy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut p = *f;
    piola(bulk_modulus, shear_modulus, f, &mut p);
    let jacobian = determinant(f);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] =
                (p[3 * i] * f[3 * j] + p[3 * i + 1] * f[3 * j + 1] + p[3 * i + 2] * f[3 * j + 2])
                    / jacobian;
        }
    }
}

/// `S = F^-1 P`.
#[autodiff_forward(d_second_piola, Const, Const, Dual, Dual)]
fn second_piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut p = *f;
    piola(bulk_modulus, shear_modulus, f, &mut p);
    let jacobian = determinant(f);
    let inverse = [
        (f[4] * f[8] - f[5] * f[7]) / jacobian,
        (f[2] * f[7] - f[1] * f[8]) / jacobian,
        (f[1] * f[5] - f[2] * f[4]) / jacobian,
        (f[5] * f[6] - f[3] * f[8]) / jacobian,
        (f[0] * f[8] - f[2] * f[6]) / jacobian,
        (f[2] * f[3] - f[0] * f[5]) / jacobian,
        (f[3] * f[7] - f[4] * f[6]) / jacobian,
        (f[1] * f[6] - f[0] * f[7]) / jacobian,
        (f[0] * f[4] - f[1] * f[3]) / jacobian,
    ];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = inverse[3 * i] * p[j]
                + inverse[3 * i + 1] * p[3 + j]
                + inverse[3 * i + 2] * p[6 + j];
        }
    }
}
