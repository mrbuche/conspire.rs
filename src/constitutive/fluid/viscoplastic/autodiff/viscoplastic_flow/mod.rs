#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

use super::AutodiffViscoplastic;
use crate::{
    math::{Quantity, Scalar},
    units::{Rate, Stress},
};
use std::autodiff::autodiff_reverse;

#[derive(Clone, Debug)]
pub struct AutodiffViscoplasticFlow {
    /// The initial yield stress.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope.
    pub hardening_slope: Quantity<Stress>,
    /// The rate sensitivity parameter.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate.
    pub reference_flow_rate: Quantity<Rate>,
}

impl AutodiffViscoplastic for AutodiffViscoplasticFlow {
    fn parameters(&self) -> [f64; 2] {
        [self.reference_flow_rate.value(), self.rate_sensitivity]
    }
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.reference_flow_rate
    }
    fn dissipation(p: &[f64; 2], plastic_stretching_rate: &[f64; 9], yield_stress: f64) -> f64 {
        dissipation(p[0], p[1], plastic_stretching_rate, yield_stress)
    }
    fn dual_dissipation(p: &[f64; 2], mandel_deviatoric: &[f64; 9], yield_stress: f64) -> f64 {
        dual_dissipation(p[0], p[1], mandel_deviatoric, yield_stress)
    }
    fn stretching_rate(
        p: &[f64; 2],
        mandel_deviatoric: &[f64; 9],
        yield_stress: f64,
        out: &mut [f64; 9],
    ) {
        for out_i in out.iter_mut() {
            *out_i = 0.0;
        }
        d_dual_dissipation(p[0], p[1], mandel_deviatoric, out, yield_stress, 1.0);
    }
}

fn dissipation(
    reference_flow_rate: f64,
    rate_sensitivity: f64,
    plastic_stretching_rate: &[f64; 9],
    yield_stress: f64,
) -> f64 {
    let mut norm_squared = 0.0;
    for k in 0..9 {
        norm_squared += plastic_stretching_rate[k] * plastic_stretching_rate[k];
    }
    let norm = norm_squared.sqrt();
    reference_flow_rate * yield_stress / (1.0 + rate_sensitivity)
        * (norm / reference_flow_rate).powf(1.0 + rate_sensitivity)
}

#[autodiff_reverse(d_dual_dissipation, Const, Const, Duplicated, Const, Active)]
fn dual_dissipation(
    reference_flow_rate: f64,
    rate_sensitivity: f64,
    mandel_deviatoric: &[f64; 9],
    yield_stress: f64,
) -> f64 {
    let mut norm_squared = 0.0;
    for k in 0..9 {
        norm_squared += mandel_deviatoric[k] * mandel_deviatoric[k];
    }
    let norm = norm_squared.sqrt();
    reference_flow_rate * yield_stress * rate_sensitivity / (1.0 + rate_sensitivity)
        * (norm / yield_stress).powf((1.0 + rate_sensitivity) / rate_sensitivity)
}
