#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

use super::AutodiffHyperviscous;
use crate::{
    constitutive::solid::autodiff::{inverse, push_cauchy, push_second_piola},
    math::Quantity,
    units::Viscosity,
};
use std::autodiff::{autodiff_forward, autodiff_reverse};

#[derive(Clone, Debug)]
pub struct AutodiffNewtonian {
    /// The bulk viscosity.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl AutodiffHyperviscous for AutodiffNewtonian {
    fn parameters(&self) -> [f64; 2] {
        [self.bulk_viscosity.value(), self.shear_viscosity.value()]
    }
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
    fn dissipation(p: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9]) -> f64 {
        dissipation(p[0], p[1], f, f_dot)
    }
    fn viscous_cauchy(p: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9], out: &mut [f64; 9]) {
        cauchy(p[0], p[1], f, f_dot, out)
    }
    fn viscous_piola(p: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9], out: &mut [f64; 9]) {
        piola(p[0], p[1], f, f_dot, out)
    }
    fn viscous_second_piola(p: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9], out: &mut [f64; 9]) {
        second_piola(p[0], p[1], f, f_dot, out)
    }
    fn viscous_cauchy_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_cauchy(p[0], p[1], f, f_dot, df_dot, primal, seed)
    }
    fn viscous_piola_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_piola(p[0], p[1], f, f_dot, df_dot, primal, seed)
    }
    fn viscous_second_piola_tangent(
        p: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    ) {
        d_second_piola(p[0], p[1], f, f_dot, df_dot, primal, seed)
    }
}

#[autodiff_reverse(d_dissipation, Const, Const, Const, Duplicated, Active)]
fn dissipation(bulk_viscosity: f64, shear_viscosity: f64, f: &[f64; 9], f_dot: &[f64; 9]) -> f64 {
    let finv = inverse(f);
    let mut strain_rate_contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let mut l_ij = 0.0;
            let mut l_ji = 0.0;
            for k in 0..3 {
                l_ij += f_dot[3 * i + k] * finv[3 * k + j];
                l_ji += f_dot[3 * j + k] * finv[3 * k + i];
            }
            let d_ij = 0.5 * (l_ij + l_ji);
            strain_rate_contraction += d_ij * d_ij;
        }
    }
    let mut strain_rate_trace = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            strain_rate_trace += f_dot[3 * i + k] * finv[3 * k + i];
        }
    }
    shear_viscosity * strain_rate_contraction
        + (bulk_viscosity - 2.0 / 3.0 * shear_viscosity)
            * strain_rate_trace
            * strain_rate_trace
            * 0.5
}

#[autodiff_forward(d_piola, Const, Const, Const, Dual, Dual)]
fn piola(
    bulk_viscosity: f64,
    shear_viscosity: f64,
    f: &[f64; 9],
    f_dot: &[f64; 9],
    out: &mut [f64; 9],
) {
    for out_i in out.iter_mut() {
        *out_i = 0.0;
    }
    d_dissipation(bulk_viscosity, shear_viscosity, f, f_dot, out, 1.0);
}

#[autodiff_forward(d_cauchy, Const, Const, Const, Dual, Dual)]
fn cauchy(
    bulk_viscosity: f64,
    shear_viscosity: f64,
    f: &[f64; 9],
    f_dot: &[f64; 9],
    out: &mut [f64; 9],
) {
    let mut p = *f_dot;
    piola(bulk_viscosity, shear_viscosity, f, f_dot, &mut p);
    push_cauchy(&p, f, out);
}

#[autodiff_forward(d_second_piola, Const, Const, Const, Dual, Dual)]
fn second_piola(
    bulk_viscosity: f64,
    shear_viscosity: f64,
    f: &[f64; 9],
    f_dot: &[f64; 9],
    out: &mut [f64; 9],
) {
    let mut p = *f_dot;
    piola(bulk_viscosity, shear_viscosity, f, f_dot, &mut p);
    push_second_piola(&p, f, out);
}
