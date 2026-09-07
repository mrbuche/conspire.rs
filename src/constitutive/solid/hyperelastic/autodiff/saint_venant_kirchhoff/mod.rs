#[cfg(test)]
mod test;

use super::{AutodiffElastic, AutodiffHyperelastic};
use crate::{
    constitutive::solid::autodiff::{push_cauchy, push_second_piola},
    math::Quantity,
    units::Stress,
};
use std::autodiff::{autodiff_forward, autodiff_reverse};

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

impl AutodiffHyperelastic for AutodiffSaintVenantKirchhoff {
    fn energy(p: &[f64; 2], f: &[f64; 9]) -> f64 {
        energy(p[0], p[1], f)
    }
}

#[autodiff_reverse(d_energy, Const, Const, Duplicated, Active)]
fn energy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9]) -> f64 {
    let c00 = f[0] * f[0] + f[3] * f[3] + f[6] * f[6];
    let c11 = f[1] * f[1] + f[4] * f[4] + f[7] * f[7];
    let c22 = f[2] * f[2] + f[5] * f[5] + f[8] * f[8];
    let c01 = f[0] * f[1] + f[3] * f[4] + f[6] * f[7];
    let c02 = f[0] * f[2] + f[3] * f[5] + f[6] * f[8];
    let c12 = f[1] * f[2] + f[4] * f[5] + f[7] * f[8];
    let trace_c = c00 + c11 + c22;
    let trace_e = 0.5 * (trace_c - 3.0);
    let squared_trace_e = 0.25
        * (c00 * c00 + c11 * c11 + c22 * c22 + 2.0 * (c01 * c01 + c02 * c02 + c12 * c12)
            - 2.0 * trace_c
            + 3.0);
    shear_modulus * squared_trace_e
        + 0.5 * (bulk_modulus - 2.0 / 3.0 * shear_modulus) * trace_e * trace_e
}

#[autodiff_forward(d_piola, Const, Const, Dual, Dual)]
fn piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    for out_i in out.iter_mut() {
        *out_i = 0.0;
    }
    d_energy(bulk_modulus, shear_modulus, f, out, 1.0);
}

#[autodiff_forward(d_cauchy, Const, Const, Dual, Dual)]
fn cauchy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut p = *f;
    piola(bulk_modulus, shear_modulus, f, &mut p);
    push_cauchy(&p, f, out);
}

#[autodiff_forward(d_second_piola, Const, Const, Dual, Dual)]
fn second_piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    let mut p = *f;
    piola(bulk_modulus, shear_modulus, f, &mut p);
    push_second_piola(&p, f, out);
}
