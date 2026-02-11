#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, IntegrationError, OdeIntegrator, VariableStep, VariableStepExplicit},
    interpolate::InterpolateSolution,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

pub const C_2: Scalar = 0.03462;
pub const C_3: Scalar = 0.097_024_350_638_780_44;
pub const C_4: Scalar = 0.145_536_525_958_170_67;
pub const C_5: Scalar = 0.561;
pub const C_6: Scalar = 0.229_007_911_590_485;
pub const C_7: Scalar = 0.544_992_088_409_515;
pub const C_8: Scalar = 0.645;
pub const C_9: Scalar = 0.48375;
pub const C_10: Scalar = 0.06757;
pub const C_11: Scalar = 0.2500;
pub const C_12: Scalar = 0.659_065_061_873_099_9;
pub const C_13: Scalar = 0.8206;
pub const C_14: Scalar = 0.9012;

pub const A_2_1: Scalar = 0.03462;
pub const A_3_1: Scalar = -0.03893354388572875;
pub const A_3_2: Scalar = 0.13595789452450918;
pub const A_4_1: Scalar = 0.03638413148954267;
pub const A_4_3: Scalar = 0.10915239446862801;
pub const A_5_1: Scalar = 2.0257639143939694;
pub const A_5_3: Scalar = -7.638023836496291;
pub const A_5_4: Scalar = 6.173259922102322;
pub const A_6_1: Scalar = 0.05112275589406061;
pub const A_6_4: Scalar = 0.17708237945550218;
pub const A_6_5: Scalar = 0.0008027762409222536;
pub const A_7_1: Scalar = 0.13160063579752163;
pub const A_7_4: Scalar = -0.2957276252669636;
pub const A_7_5: Scalar = 0.08781378035642955;
pub const A_7_6: Scalar = 0.6213052975225274;
pub const A_8_1: Scalar = 0.07166666666666667;
pub const A_8_6: Scalar = 0.33055335789153195;
pub const A_8_7: Scalar = 0.2427799754418014;
pub const A_9_1: Scalar = 0.071806640625;
pub const A_9_6: Scalar = 0.3294380283228177;
pub const A_9_7: Scalar = 0.1165190029271823;
pub const A_9_8: Scalar = -0.034013671875;
pub const A_10_1: Scalar = 0.04836757646340646;
pub const A_10_6: Scalar = 0.03928989925676164;
pub const A_10_7: Scalar = 0.10547409458903446;
pub const A_10_8: Scalar = -0.021438652846483126;
pub const A_10_9: Scalar = -0.10412291746271944;
pub const A_11_1: Scalar = -0.026645614872014785;
pub const A_11_6: Scalar = 0.03333333333333333;
pub const A_11_7: Scalar = -0.1631072244872467;
pub const A_11_8: Scalar = 0.03396081684127761;
pub const A_11_9: Scalar = 0.1572319413814626;
pub const A_11_10: Scalar = 0.21522674780318796;
pub const A_12_1: Scalar = 0.03689009248708622;
pub const A_12_6: Scalar = -0.1465181576725543;
pub const A_12_7: Scalar = 0.2242577768172024;
pub const A_12_8: Scalar = 0.02294405717066073;
pub const A_12_9: Scalar = -0.0035850052905728597;
pub const A_12_10: Scalar = 0.08669223316444385;
pub const A_12_11: Scalar = 0.43838406519683376;
pub const A_13_1: Scalar = -0.4866012215113341;
pub const A_13_6: Scalar = -6.304602650282853;
pub const A_13_7: Scalar = -0.2812456182894729;
pub const A_13_8: Scalar = -2.679019236219849;
pub const A_13_9: Scalar = 0.5188156639241577;
pub const A_13_10: Scalar = 1.3653531876033418;
pub const A_13_11: Scalar = 5.8850910885039465;
pub const A_13_12: Scalar = 2.8028087862720628;
pub const A_14_1: Scalar = 0.4185367457753472;
pub const A_14_6: Scalar = 6.724547581906459;
pub const A_14_7: Scalar = -0.42544428016461133;
pub const A_14_8: Scalar = 3.3432791530012653;
pub const A_14_9: Scalar = 0.6170816631175374;
pub const A_14_10: Scalar = -0.9299661239399329;
pub const A_14_11: Scalar = -6.099948804751011;
pub const A_14_12: Scalar = -3.002206187889399;
pub const A_14_13: Scalar = 0.2553202529443446;
pub const A_15_1: Scalar = -0.7793740861228848;
pub const A_15_6: Scalar = -13.937342538107776;
pub const A_15_7: Scalar = 1.2520488533793563;
pub const A_15_8: Scalar = -14.691500408016868;
pub const A_15_9: Scalar = -0.494705058533141;
pub const A_15_10: Scalar = 2.2429749091462368;
pub const A_15_11: Scalar = 13.367893803828643;
pub const A_15_12: Scalar = 14.396650486650687;
pub const A_15_13: Scalar = -0.79758133317768;
pub const A_15_14: Scalar = 0.4409353709534278;
pub const A_16_1: Scalar = 2.0580513374668867;
pub const A_16_6: Scalar = 22.357937727968032;
pub const A_16_7: Scalar = 0.9094981099755646;
pub const A_16_8: Scalar = 35.89110098240264;
pub const A_16_9: Scalar = -3.442515027624454;
pub const A_16_10: Scalar = -4.865481358036369;
pub const A_16_11: Scalar = -18.909803813543427;
pub const A_16_12: Scalar = -34.26354448030452;
pub const A_16_13: Scalar = 1.2647565216956427;

pub const B_1: Scalar = 0.014611976858423152;
pub const B_8: Scalar = -0.3915211862331339;
pub const B_9: Scalar = 0.23109325002895065;
pub const B_10: Scalar = 0.12747667699928525;
pub const B_11: Scalar = 0.2246434176204158;
pub const B_12: Scalar = 0.5684352689748513;
pub const B_13: Scalar = 0.058258715572158275;
pub const B_14: Scalar = 0.13643174034822156;
pub const B_15: Scalar = 0.030570139830827976;

pub const D_1: Scalar = -0.005357988290444578;
pub const D_8: Scalar = -2.583020491182464;
pub const D_9: Scalar = 0.14252253154686625;
pub const D_10: Scalar = 0.013420653512688676;
pub const D_11: Scalar = -0.02867296291409493;
pub const D_12: Scalar = 2.624999655215792;
pub const D_13: Scalar = -0.2825509643291537;
pub const D_14: Scalar = 0.13643174034822156;
pub const D_15: Scalar = 0.030570139830827976;
pub const D_16: Scalar = -0.04834231373823958;

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Verner9 {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Relative error tolerance.
    pub rel_tol: Scalar,
    /// Multiplier for adaptive time steps.
    pub dt_beta: Scalar,
    /// Exponent for adaptive time steps.
    pub dt_expn: Scalar,
    /// Cut back factor for the time step.
    pub dt_cut: Scalar,
    /// Minimum value for the time step.
    pub dt_min: Scalar,
}

impl Default for Verner9 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 9.0,
            dt_cut: 0.5,
            dt_min: ABS_TOL,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for Verner9
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl VariableStep for Verner9 {
    fn abs_tol(&self) -> Scalar {
        self.abs_tol
    }
    fn rel_tol(&self) -> Scalar {
        self.rel_tol
    }
    fn dt_beta(&self) -> Scalar {
        self.dt_beta
    }
    fn dt_expn(&self) -> Scalar {
        self.dt_expn
    }
    fn dt_cut(&self) -> Scalar {
        self.dt_cut
    }
    fn dt_min(&self) -> Scalar {
        self.dt_min
    }
}

impl<Y, U> Explicit<Y, U> for Verner9
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 16;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U> VariableStepExplicit<Y, U> for Verner9
where
    Self: Explicit<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn error(dt: Scalar, k: &[Y]) -> Result<Scalar, String> {
        Ok(((&k[0] * D_1
            + &k[7] * D_8
            + &k[8] * D_9
            + &k[9] * D_10
            + &k[10] * D_11
            + &k[11] * D_12
            + &k[12] * D_13
            + &k[13] * D_14
            + &k[14] * D_15
            + &k[15] * D_16)
            * dt)
            .norm_inf())
    }
    fn slopes(
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        k[0] = function(t, y)?;
        *y_trial = &k[0] * (A_2_1 * dt) + y;
        k[1] = function(t + C_2 * dt, y_trial)?;
        *y_trial = &k[0] * (A_3_1 * dt) + &k[1] * (A_3_2 * dt) + y;
        k[2] = function(t + C_3 * dt, y_trial)?;
        *y_trial = &k[0] * (A_4_1 * dt) + &k[2] * (A_4_3 * dt) + y;
        k[3] = function(t + C_4 * dt, y_trial)?;
        *y_trial = &k[0] * (A_5_1 * dt) + &k[2] * (A_5_3 * dt) + &k[3] * (A_5_4 * dt) + y;
        k[4] = function(t + C_5 * dt, y_trial)?;
        *y_trial = &k[0] * (A_6_1 * dt) + &k[3] * (A_6_4 * dt) + &k[4] * (A_6_5 * dt) + y;
        k[5] = function(t + C_6 * dt, y_trial)?;
        *y_trial = &k[0] * (A_7_1 * dt)
            + &k[3] * (A_7_4 * dt)
            + &k[4] * (A_7_5 * dt)
            + &k[5] * (A_7_6 * dt)
            + y;
        k[6] = function(t + C_7 * dt, y_trial)?;
        *y_trial = &k[0] * (A_8_1 * dt) + &k[5] * (A_8_6 * dt) + &k[6] * (A_8_7 * dt) + y;
        k[7] = function(t + C_8 * dt, y_trial)?;
        *y_trial = &k[0] * (A_9_1 * dt)
            + &k[5] * (A_9_6 * dt)
            + &k[6] * (A_9_7 * dt)
            + &k[7] * (A_9_8 * dt)
            + y;
        k[8] = function(t + C_9 * dt, y_trial)?;
        *y_trial = &k[0] * (A_10_1 * dt)
            + &k[5] * (A_10_6 * dt)
            + &k[6] * (A_10_7 * dt)
            + &k[7] * (A_10_8 * dt)
            + &k[8] * (A_10_9 * dt)
            + y;
        k[9] = function(t + C_10 * dt, y_trial)?;
        *y_trial = &k[0] * (A_11_1 * dt)
            + &k[5] * (A_11_6 * dt)
            + &k[6] * (A_11_7 * dt)
            + &k[7] * (A_11_8 * dt)
            + &k[8] * (A_11_9 * dt)
            + &k[9] * (A_11_10 * dt)
            + y;
        k[10] = function(t + C_11 * dt, y_trial)?;
        *y_trial = &k[0] * (A_12_1 * dt)
            + &k[5] * (A_12_6 * dt)
            + &k[6] * (A_12_7 * dt)
            + &k[7] * (A_12_8 * dt)
            + &k[8] * (A_12_9 * dt)
            + &k[9] * (A_12_10 * dt)
            + &k[10] * (A_12_11 * dt)
            + y;
        k[11] = function(t + C_12 * dt, y_trial)?;
        *y_trial = &k[0] * (A_13_1 * dt)
            + &k[5] * (A_13_6 * dt)
            + &k[6] * (A_13_7 * dt)
            + &k[7] * (A_13_8 * dt)
            + &k[8] * (A_13_9 * dt)
            + &k[9] * (A_13_10 * dt)
            + &k[10] * (A_13_11 * dt)
            + &k[11] * (A_13_12 * dt)
            + y;
        k[12] = function(t + C_13 * dt, y_trial)?;
        *y_trial = &k[0] * (A_14_1 * dt)
            + &k[5] * (A_14_6 * dt)
            + &k[6] * (A_14_7 * dt)
            + &k[7] * (A_14_8 * dt)
            + &k[8] * (A_14_9 * dt)
            + &k[9] * (A_14_10 * dt)
            + &k[10] * (A_14_11 * dt)
            + &k[11] * (A_14_12 * dt)
            + &k[12] * (A_14_13 * dt)
            + y;
        k[13] = function(t + C_14 * dt, y_trial)?;
        *y_trial = &k[0] * (A_15_1 * dt)
            + &k[5] * (A_15_6 * dt)
            + &k[6] * (A_15_7 * dt)
            + &k[7] * (A_15_8 * dt)
            + &k[8] * (A_15_9 * dt)
            + &k[9] * (A_15_10 * dt)
            + &k[10] * (A_15_11 * dt)
            + &k[11] * (A_15_12 * dt)
            + &k[12] * (A_15_13 * dt)
            + &k[13] * (A_15_14 * dt)
            + y;
        k[14] = function(t + dt, y_trial)?;
        *y_trial = &k[0] * (A_16_1 * dt)
            + &k[5] * (A_16_6 * dt)
            + &k[6] * (A_16_7 * dt)
            + &k[7] * (A_16_8 * dt)
            + &k[8] * (A_16_9 * dt)
            + &k[9] * (A_16_10 * dt)
            + &k[10] * (A_16_11 * dt)
            + &k[11] * (A_16_12 * dt)
            + &k[12] * (A_16_13 * dt)
            + y;
        if k.len() == Self::SLOPES {
            k[15] = function(t + dt, y_trial)?;
        }
        *y_trial = (&k[0] * B_1
            + &k[7] * B_8
            + &k[8] * B_9
            + &k[9] * B_10
            + &k[10] * B_11
            + &k[11] * B_12
            + &k[12] * B_13
            + &k[13] * B_14
            + &k[14] * B_15)
            * dt
            + y;
        Ok(())
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Verner9
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        Self::interpolate_variable_step(time, tp, yp, function)
    }
}
