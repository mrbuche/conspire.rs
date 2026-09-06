#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

use super::{AutodiffElastic, determinant, push_first_piola, push_second_piola};
use crate::{math::Quantity, units::Stress};
use std::autodiff::autodiff_forward;

/// The elastic
/// [`SaintVenantKirchhoff`](crate::constitutive::solid::elastic::SaintVenantKirchhoff)
/// as autodiff kernels (Cauchy stress written directly on the spatial strain,
/// no potential); wrap in [`Autodiff`](super::Autodiff) for the `Elastic` API.
#[derive(Clone, Debug)]
pub struct AutodiffSaintVenantKirchhoff {
    /// The bulk modulus.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus.
    pub shear_modulus: Quantity<Stress>,
}

impl AutodiffElastic for AutodiffSaintVenantKirchhoff {
    fn parameters(&self) -> [f64; 2] {
        [self.bulk_modulus.value(), self.shear_modulus.value()]
    }
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
    fn cauchy(p: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]) {
        cauchy(p[0], p[1], f, out)
    }
    fn piola(p: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]) {
        piola(p[0], p[1], f, out)
    }
    fn second_piola(p: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]) {
        second_piola(p[0], p[1], f, out)
    }
    fn cauchy_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_cauchy(p[0], p[1], f, df, primal, seed)
    }
    fn piola_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_piola(p[0], p[1], f, df, primal, seed)
    }
    fn second_piola_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_second_piola(p[0], p[1], f, df, primal, seed)
    }
}

/// `sigma = J^-1 [2 mu eps + (kappa - 2 mu / 3) tr(eps) I]`,
/// `eps = (F F^T - I) / 2`, row-major.
///
/// Mirrors `<SaintVenantKirchhoff as Elastic>::cauchy_stress`.
#[autodiff_forward(d_cauchy, Const, Const, Dual, Dual)]
fn cauchy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let strain = [
        0.5 * (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] - 1.0),
        0.5 * (f[0] * f[3] + f[1] * f[4] + f[2] * f[5]),
        0.5 * (f[0] * f[6] + f[1] * f[7] + f[2] * f[8]),
        0.5 * (f[3] * f[0] + f[4] * f[1] + f[5] * f[2]),
        0.5 * (f[3] * f[3] + f[4] * f[4] + f[5] * f[5] - 1.0),
        0.5 * (f[3] * f[6] + f[4] * f[7] + f[5] * f[8]),
        0.5 * (f[6] * f[0] + f[7] * f[1] + f[8] * f[2]),
        0.5 * (f[6] * f[3] + f[7] * f[4] + f[8] * f[5]),
        0.5 * (f[6] * f[6] + f[7] * f[7] + f[8] * f[8] - 1.0),
    ];
    let trace = strain[0] + strain[4] + strain[8];
    let jacobian = determinant(f);
    let two_mu = 2.0 * shear_modulus / jacobian;
    let lambda_trace = (bulk_modulus - 2.0 / 3.0 * shear_modulus) * trace / jacobian;
    for i in 0..9 {
        out[i] = two_mu * strain[i];
    }
    out[0] += lambda_trace;
    out[4] += lambda_trace;
    out[8] += lambda_trace;
}

/// `P = J sigma F^-T`.
#[autodiff_forward(d_piola, Const, Const, Dual, Dual)]
fn piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut sigma = *f;
    cauchy(bulk_modulus, shear_modulus, f, &mut sigma);
    push_first_piola(&sigma, f, out);
}

/// `S = F^-1 P`.
#[autodiff_forward(d_second_piola, Const, Const, Dual, Dual)]
fn second_piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut p = *f;
    piola(bulk_modulus, shear_modulus, f, &mut p);
    push_second_piola(&p, f, out);
}
