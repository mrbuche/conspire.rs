#[cfg(test)]
mod test;

use super::{
    super::{
        Tensor, TensorArray, TensorRank0, TensorVec, Vector, interpolate::InterpolateSolution,
    },
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_2: TensorRank0 = 0.03462;
const C_3: TensorRank0 = 0.097_024_350_638_780_44;
const C_4: TensorRank0 = 0.145_536_525_958_170_67;
const C_5: TensorRank0 = 0.561;
const C_6: TensorRank0 = 0.229_007_911_590_485;
const C_7: TensorRank0 = 0.544_992_088_409_515;
const C_8: TensorRank0 = 0.645;
const C_9: TensorRank0 = 0.48375;
const C_10: TensorRank0 = 0.06757;
const C_11: TensorRank0 = 0.2500;
const C_12: TensorRank0 = 0.659_065_061_873_099_9;
const C_13: TensorRank0 = 0.8206;
const C_14: TensorRank0 = 0.9012;

const A_2_1: TensorRank0 = 0.03462;
const A_3_1: TensorRank0 = -0.03893354388572875;
const A_3_2: TensorRank0 = 0.13595789452450918;
const A_4_1: TensorRank0 = 0.03638413148954267;
const A_4_3: TensorRank0 = 0.10915239446862801;
const A_5_1: TensorRank0 = 2.0257639143939694;
const A_5_3: TensorRank0 = -7.638023836496291;
const A_5_4: TensorRank0 = 6.173259922102322;
const A_6_1: TensorRank0 = 0.05112275589406061;
const A_6_4: TensorRank0 = 0.17708237945550218;
const A_6_5: TensorRank0 = 0.0008027762409222536;
const A_7_1: TensorRank0 = 0.13160063579752163;
const A_7_4: TensorRank0 = -0.2957276252669636;
const A_7_5: TensorRank0 = 0.08781378035642955;
const A_7_6: TensorRank0 = 0.6213052975225274;
const A_8_1: TensorRank0 = 0.07166666666666667;
const A_8_6: TensorRank0 = 0.33055335789153195;
const A_8_7: TensorRank0 = 0.2427799754418014;
const A_9_1: TensorRank0 = 0.071806640625;
const A_9_6: TensorRank0 = 0.3294380283228177;
const A_9_7: TensorRank0 = 0.1165190029271823;
const A_9_8: TensorRank0 = -0.034013671875;
const A_10_1: TensorRank0 = 0.04836757646340646;
const A_10_6: TensorRank0 = 0.03928989925676164;
const A_10_7: TensorRank0 = 0.10547409458903446;
const A_10_8: TensorRank0 = -0.021438652846483126;
const A_10_9: TensorRank0 = -0.10412291746271944;
const A_11_1: TensorRank0 = -0.026645614872014785;
const A_11_6: TensorRank0 = 0.03333333333333333;
const A_11_7: TensorRank0 = -0.1631072244872467;
const A_11_8: TensorRank0 = 0.03396081684127761;
const A_11_9: TensorRank0 = 0.1572319413814626;
const A_11_10: TensorRank0 = 0.21522674780318796;
const A_12_1: TensorRank0 = 0.03689009248708622;
const A_12_6: TensorRank0 = -0.1465181576725543;
const A_12_7: TensorRank0 = 0.2242577768172024;
const A_12_8: TensorRank0 = 0.02294405717066073;
const A_12_9: TensorRank0 = -0.0035850052905728597;
const A_12_10: TensorRank0 = 0.08669223316444385;
const A_12_11: TensorRank0 = 0.43838406519683376;
const A_13_1: TensorRank0 = -0.4866012215113341;
const A_13_6: TensorRank0 = -6.304602650282853;
const A_13_7: TensorRank0 = -0.2812456182894729;
const A_13_8: TensorRank0 = -2.679019236219849;
const A_13_9: TensorRank0 = 0.5188156639241577;
const A_13_10: TensorRank0 = 1.3653531876033418;
const A_13_11: TensorRank0 = 5.8850910885039465;
const A_13_12: TensorRank0 = 2.8028087862720628;
const A_14_1: TensorRank0 = 0.4185367457753472;
const A_14_6: TensorRank0 = 6.724547581906459;
const A_14_7: TensorRank0 = -0.42544428016461133;
const A_14_8: TensorRank0 = 3.3432791530012653;
const A_14_9: TensorRank0 = 0.6170816631175374;
const A_14_10: TensorRank0 = -0.9299661239399329;
const A_14_11: TensorRank0 = -6.099948804751011;
const A_14_12: TensorRank0 = -3.002206187889399;
const A_14_13: TensorRank0 = 0.2553202529443446;
const A_15_1: TensorRank0 = -0.7793740861228848;
const A_15_6: TensorRank0 = -13.937342538107776;
const A_15_7: TensorRank0 = 1.2520488533793563;
const A_15_8: TensorRank0 = -14.691500408016868;
const A_15_9: TensorRank0 = -0.494705058533141;
const A_15_10: TensorRank0 = 2.2429749091462368;
const A_15_11: TensorRank0 = 13.367893803828643;
const A_15_12: TensorRank0 = 14.396650486650687;
const A_15_13: TensorRank0 = -0.79758133317768;
const A_15_14: TensorRank0 = 0.4409353709534278;
const A_16_1: TensorRank0 = 2.0580513374668867;
const A_16_6: TensorRank0 = 22.357937727968032;
const A_16_7: TensorRank0 = 0.9094981099755646;
const A_16_8: TensorRank0 = 35.89110098240264;
const A_16_9: TensorRank0 = -3.442515027624454;
const A_16_10: TensorRank0 = -4.865481358036369;
const A_16_11: TensorRank0 = -18.909803813543427;
const A_16_12: TensorRank0 = -34.26354448030452;
const A_16_13: TensorRank0 = 1.2647565216956427;

const B_1: TensorRank0 = 0.014611976858423152;
const B_8: TensorRank0 = -0.3915211862331339;
const B_9: TensorRank0 = 0.23109325002895065;
const B_10: TensorRank0 = 0.12747667699928525;
const B_11: TensorRank0 = 0.2246434176204158;
const B_12: TensorRank0 = 0.5684352689748513;
const B_13: TensorRank0 = 0.058258715572158275;
const B_14: TensorRank0 = 0.13643174034822156;
const B_15: TensorRank0 = 0.030570139830827976;

const D_1: TensorRank0 = -0.005357988290444578;
const D_8: TensorRank0 = -2.583020491182464;
const D_9: TensorRank0 = 0.14252253154686625;
const D_10: TensorRank0 = 0.013420653512688676;
const D_11: TensorRank0 = -0.02867296291409493;
const D_12: TensorRank0 = 2.624999655215792;
const D_13: TensorRank0 = -0.2825509643291537;
const D_14: TensorRank0 = 0.13643174034822156;
const D_15: TensorRank0 = 0.030570139830827976;
const D_16: TensorRank0 = -0.04834231373823958;

/// Explicit, sixteen-stage, ninth-order, variable-step, Runge-Kutta method.[^cite]
///
/// [^cite]: J.H. Verner, [Numer. Algor. **53**, 383 (2010)](https://doi.org/10.1007/s11075-009-9290-3).
///
/// ```math
/// \frac{dy}{dt} = f(t, y)
/// ```
/// ```math
/// t_{n+1} = t_n + h
/// ```
/// ```math
/// k_1 = f(t_n, y_n)
/// ```
/// ```math
/// \cdots
/// ```
/// ```math
/// h_{n+1} = \beta h \left(\frac{e_\mathrm{tol}}{e_{n+1}}\right)^{1/p}
/// ```

#[derive(Debug)]
pub struct Verner9 {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
    /// Multiplier for adaptive time steps.
    pub dt_beta: TensorRank0,
    /// Exponent for adaptive time steps.
    pub dt_expn: TensorRank0,
    /// Initial relative time step.
    pub dt_init: TensorRank0,
}

impl Default for Verner9 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 9.0,
            dt_init: 0.1,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Verner9
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        time: &[TensorRank0],
        initial_condition: Y,
    ) -> Result<(Vector, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = time[0];
        let mut dt = self.dt_init * time[time.len() - 1];
        let mut e;
        let mut k_1;
        let mut k_2;
        let mut k_3;
        let mut k_4;
        let mut k_5;
        let mut k_6;
        let mut k_7;
        let mut k_8;
        let mut k_9;
        let mut k_10;
        let mut k_11;
        let mut k_12;
        let mut k_13;
        let mut k_14;
        let mut k_15;
        let mut k_16;
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut y = initial_condition.clone();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.clone());
        let mut y_trial;
        while t < time[time.len() - 1] {
            k_1 = function(&t, &y);
            k_2 = function(&(t + C_2 * dt), &(&k_1 * (A_2_1 * dt) + &y));
            k_3 = function(
                &(t + C_3 * dt),
                &(&k_1 * (A_3_1 * dt) + &k_2 * (A_3_2 * dt) + &y),
            );
            k_4 = function(
                &(t + C_4 * dt),
                &(&k_1 * (A_4_1 * dt) + &k_3 * (A_4_3 * dt) + &y),
            );
            k_5 = function(
                &(t + C_5 * dt),
                &(&k_1 * (A_5_1 * dt) + &k_3 * (A_5_3 * dt) + &k_4 * (A_5_4 * dt) + &y),
            );
            k_6 = function(
                &(t + C_6 * dt),
                &(&k_1 * (A_6_1 * dt) + &k_4 * (A_6_4 * dt) + &k_5 * (A_6_5 * dt) + &y),
            );
            k_7 = function(
                &(t + C_7 * dt),
                &(&k_1 * (A_7_1 * dt)
                    + &k_4 * (A_7_4 * dt)
                    + &k_5 * (A_7_5 * dt)
                    + &k_6 * (A_7_6 * dt)
                    + &y),
            );
            k_8 = function(
                &(t + C_8 * dt),
                &(&k_1 * (A_8_1 * dt) + &k_6 * (A_8_6 * dt) + &k_7 * (A_8_7 * dt) + &y),
            );
            k_9 = function(
                &(t + C_9 * dt),
                &(&k_1 * (A_9_1 * dt)
                    + &k_6 * (A_9_6 * dt)
                    + &k_7 * (A_9_7 * dt)
                    + &k_8 * (A_9_8 * dt)
                    + &y),
            );
            k_10 = function(
                &(t + C_10 * dt),
                &(&k_1 * (A_10_1 * dt)
                    + &k_6 * (A_10_6 * dt)
                    + &k_7 * (A_10_7 * dt)
                    + &k_8 * (A_10_8 * dt)
                    + &k_9 * (A_10_9 * dt)
                    + &y),
            );
            k_11 = function(
                &(t + C_11 * dt),
                &(&k_1 * (A_11_1 * dt)
                    + &k_6 * (A_11_6 * dt)
                    + &k_7 * (A_11_7 * dt)
                    + &k_8 * (A_11_8 * dt)
                    + &k_9 * (A_11_9 * dt)
                    + &k_10 * (A_11_10 * dt)
                    + &y),
            );
            k_12 = function(
                &(t + C_12 * dt),
                &(&k_1 * (A_12_1 * dt)
                    + &k_6 * (A_12_6 * dt)
                    + &k_7 * (A_12_7 * dt)
                    + &k_8 * (A_12_8 * dt)
                    + &k_9 * (A_12_9 * dt)
                    + &k_10 * (A_12_10 * dt)
                    + &k_11 * (A_12_11 * dt)
                    + &y),
            );
            k_13 = function(
                &(t + C_13 * dt),
                &(&k_1 * (A_13_1 * dt)
                    + &k_6 * (A_13_6 * dt)
                    + &k_7 * (A_13_7 * dt)
                    + &k_8 * (A_13_8 * dt)
                    + &k_9 * (A_13_9 * dt)
                    + &k_10 * (A_13_10 * dt)
                    + &k_11 * (A_13_11 * dt)
                    + &k_12 * (A_13_12 * dt)
                    + &y),
            );
            k_14 = function(
                &(t + C_14 * dt),
                &(&k_1 * (A_14_1 * dt)
                    + &k_6 * (A_14_6 * dt)
                    + &k_7 * (A_14_7 * dt)
                    + &k_8 * (A_14_8 * dt)
                    + &k_9 * (A_14_9 * dt)
                    + &k_10 * (A_14_10 * dt)
                    + &k_11 * (A_14_11 * dt)
                    + &k_12 * (A_14_12 * dt)
                    + &k_13 * (A_14_13 * dt)
                    + &y),
            );
            k_15 = function(
                &(t + dt),
                &(&k_1 * (A_15_1 * dt)
                    + &k_6 * (A_15_6 * dt)
                    + &k_7 * (A_15_7 * dt)
                    + &k_8 * (A_15_8 * dt)
                    + &k_9 * (A_15_9 * dt)
                    + &k_10 * (A_15_10 * dt)
                    + &k_11 * (A_15_11 * dt)
                    + &k_12 * (A_15_12 * dt)
                    + &k_13 * (A_15_13 * dt)
                    + &k_14 * (A_15_14 * dt)
                    + &y),
            );
            y_trial = (&k_1 * B_1
                + &k_8 * B_8
                + &k_9 * B_9
                + &k_10 * B_10
                + &k_11 * B_11
                + &k_12 * B_12
                + &k_13 * B_13
                + &k_14 * B_14
                + &k_15 * B_15)
                * dt
                + &y;
            k_16 = function(
                &(t + dt),
                &(&k_1 * (A_16_1 * dt)
                    + &k_6 * (A_16_6 * dt)
                    + &k_7 * (A_16_7 * dt)
                    + &k_8 * (A_16_8 * dt)
                    + &k_9 * (A_16_9 * dt)
                    + &k_10 * (A_16_10 * dt)
                    + &k_11 * (A_16_11 * dt)
                    + &k_12 * (A_16_12 * dt)
                    + &k_13 * (A_16_13 * dt)
                    + &y),
            );
            e = ((&k_1 * D_1
                + &k_8 * D_8
                + &k_9 * D_9
                + &k_10 * D_10
                + &k_11 * D_11
                + &k_12 * D_12
                + &k_13 * D_13
                + &k_14 * D_14
                + &k_15 * D_15
                + &k_16 * D_16)
                * dt)
                .norm();
            if e < self.abs_tol || e / y_trial.norm() < self.rel_tol {
                t += dt;
                y = y_trial;
                t_sol.push(t);
                y_sol.push(y.clone());
            }
            dt *= self.dt_beta * (self.abs_tol / e).powf(1.0 / self.dt_expn);
        }
        if time.len() > 2 {
            let t_int = Vector::new(time);
            let y_int = self.interpolate(&t_int, &t_sol, &y_sol, function);
            Ok((t_int, y_int))
        } else {
            Ok((t_sol, y_sol))
        }
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Verner9
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        let mut dt = 0.0;
        let mut i = 0;
        let mut k_1 = Y::zero();
        let mut k_2 = Y::zero();
        let mut k_3 = Y::zero();
        let mut k_4 = Y::zero();
        let mut k_5 = Y::zero();
        let mut k_6 = Y::zero();
        let mut k_7 = Y::zero();
        let mut k_8 = Y::zero();
        let mut k_9 = Y::zero();
        let mut k_10 = Y::zero();
        let mut k_11 = Y::zero();
        let mut k_12 = Y::zero();
        let mut k_13 = Y::zero();
        let mut k_14 = Y::zero();
        let mut k_15 = Y::zero();
        let mut t = 0.0;
        let mut y = Y::zero();
        time.iter()
            .map(|time_k| {
                i = tp.iter().position(|tp_i| tp_i > time_k).unwrap();
                t = tp[i - 1];
                y = yp[i - 1].clone();
                dt = time_k - t;
                k_1 = function(&t, &y);
                k_2 = function(&(t + C_2 * dt), &(&k_1 * (A_2_1 * dt) + &y));
                k_3 = function(
                    &(t + C_3 * dt),
                    &(&k_1 * (A_3_1 * dt) + &k_2 * (A_3_2 * dt) + &y),
                );
                k_4 = function(
                    &(t + C_4 * dt),
                    &(&k_1 * (A_4_1 * dt) + &k_3 * (A_4_3 * dt) + &y),
                );
                k_5 = function(
                    &(t + C_5 * dt),
                    &(&k_1 * (A_5_1 * dt) + &k_3 * (A_5_3 * dt) + &k_4 * (A_5_4 * dt) + &y),
                );
                k_6 = function(
                    &(t + C_6 * dt),
                    &(&k_1 * (A_6_1 * dt) + &k_4 * (A_6_4 * dt) + &k_5 * (A_6_5 * dt) + &y),
                );
                k_7 = function(
                    &(t + C_7 * dt),
                    &(&k_1 * (A_7_1 * dt)
                        + &k_4 * (A_7_4 * dt)
                        + &k_5 * (A_7_5 * dt)
                        + &k_6 * (A_7_6 * dt)
                        + &y),
                );
                k_8 = function(
                    &(t + C_8 * dt),
                    &(&k_1 * (A_8_1 * dt) + &k_6 * (A_8_6 * dt) + &k_7 * (A_8_7 * dt) + &y),
                );
                k_9 = function(
                    &(t + C_9 * dt),
                    &(&k_1 * (A_9_1 * dt)
                        + &k_6 * (A_9_6 * dt)
                        + &k_7 * (A_9_7 * dt)
                        + &k_8 * (A_9_8 * dt)
                        + &y),
                );
                k_10 = function(
                    &(t + C_10 * dt),
                    &(&k_1 * (A_10_1 * dt)
                        + &k_6 * (A_10_6 * dt)
                        + &k_7 * (A_10_7 * dt)
                        + &k_8 * (A_10_8 * dt)
                        + &k_9 * (A_10_9 * dt)
                        + &y),
                );
                k_11 = function(
                    &(t + C_11 * dt),
                    &(&k_1 * (A_11_1 * dt)
                        + &k_6 * (A_11_6 * dt)
                        + &k_7 * (A_11_7 * dt)
                        + &k_8 * (A_11_8 * dt)
                        + &k_9 * (A_11_9 * dt)
                        + &k_10 * (A_11_10 * dt)
                        + &y),
                );
                k_12 = function(
                    &(t + C_12 * dt),
                    &(&k_1 * (A_12_1 * dt)
                        + &k_6 * (A_12_6 * dt)
                        + &k_7 * (A_12_7 * dt)
                        + &k_8 * (A_12_8 * dt)
                        + &k_9 * (A_12_9 * dt)
                        + &k_10 * (A_12_10 * dt)
                        + &k_11 * (A_12_11 * dt)
                        + &y),
                );
                k_13 = function(
                    &(t + C_13 * dt),
                    &(&k_1 * (A_13_1 * dt)
                        + &k_6 * (A_13_6 * dt)
                        + &k_7 * (A_13_7 * dt)
                        + &k_8 * (A_13_8 * dt)
                        + &k_9 * (A_13_9 * dt)
                        + &k_10 * (A_13_10 * dt)
                        + &k_11 * (A_13_11 * dt)
                        + &k_12 * (A_13_12 * dt)
                        + &y),
                );
                k_14 = function(
                    &(t + C_14 * dt),
                    &(&k_1 * (A_14_1 * dt)
                        + &k_6 * (A_14_6 * dt)
                        + &k_7 * (A_14_7 * dt)
                        + &k_8 * (A_14_8 * dt)
                        + &k_9 * (A_14_9 * dt)
                        + &k_10 * (A_14_10 * dt)
                        + &k_11 * (A_14_11 * dt)
                        + &k_12 * (A_14_12 * dt)
                        + &k_13 * (A_14_13 * dt)
                        + &y),
                );
                k_15 = function(
                    &(t + dt),
                    &(&k_1 * (A_15_1 * dt)
                        + &k_6 * (A_15_6 * dt)
                        + &k_7 * (A_15_7 * dt)
                        + &k_8 * (A_15_8 * dt)
                        + &k_9 * (A_15_9 * dt)
                        + &k_10 * (A_15_10 * dt)
                        + &k_11 * (A_15_11 * dt)
                        + &k_12 * (A_15_12 * dt)
                        + &k_13 * (A_15_13 * dt)
                        + &k_14 * (A_15_14 * dt)
                        + &y),
                );
                (&k_1 * B_1
                    + &k_8 * B_8
                    + &k_9 * B_9
                    + &k_10 * B_10
                    + &k_11 * B_11
                    + &k_12 * B_12
                    + &k_13 * B_13
                    + &k_14 * B_14
                    + &k_15 * B_15)
                    * dt
                    + &y
            })
            .collect()
    }
}
