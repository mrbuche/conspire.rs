#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, IntegrationError, OdeIntegrator, VariableStep, VariableStepExplicit},
    interpolate::InterpolateSolution,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

pub const C_2: Scalar = 0.05;
pub const C_3: Scalar = 0.1065625;
pub const C_4: Scalar = 0.15984375;
pub const C_5: Scalar = 0.39;
pub const C_6: Scalar = 0.465;
pub const C_7: Scalar = 0.155;
pub const C_8: Scalar = 0.943;
pub const C_9: Scalar = 0.901802041735857;
pub const C_10: Scalar = 0.909;
pub const C_11: Scalar = 0.94;

pub const A_2_1: Scalar = 0.05;
pub const A_3_1: Scalar = -0.0069931640625;
pub const A_3_2: Scalar = 0.1135556640625;
pub const A_4_1: Scalar = 0.0399609375;
pub const A_4_3: Scalar = 0.1198828125;
pub const A_5_1: Scalar = 0.36139756280045754;
pub const A_5_3: Scalar = -1.3415240667004928;
pub const A_5_4: Scalar = 1.3701265039000352;
pub const A_6_1: Scalar = 0.049047202797202795;
pub const A_6_4: Scalar = 0.23509720422144048;
pub const A_6_5: Scalar = 0.18085559298135673;
pub const A_7_1: Scalar = 0.06169289044289044;
pub const A_7_4: Scalar = 0.11236568314640277;
pub const A_7_5: Scalar = -0.03885046071451367;
pub const A_7_6: Scalar = 0.01979188712522046;
pub const A_8_1: Scalar = -1.767630240222327;
pub const A_8_4: Scalar = -62.5;
pub const A_8_5: Scalar = -6.061889377376669;
pub const A_8_6: Scalar = 5.6508231982227635;
pub const A_8_7: Scalar = 65.62169641937624;
pub const A_9_1: Scalar = -1.1809450665549708;
pub const A_9_4: Scalar = -41.50473441114321;
pub const A_9_5: Scalar = -4.434438319103725;
pub const A_9_6: Scalar = 4.260408188586133;
pub const A_9_7: Scalar = 43.75364022446172;
pub const A_9_8: Scalar = 0.00787142548991231;
pub const A_10_1: Scalar = -1.2814059994414884;
pub const A_10_4: Scalar = -45.047139960139866;
pub const A_10_5: Scalar = -4.731362069449576;
pub const A_10_6: Scalar = 4.514967016593808;
pub const A_10_7: Scalar = 47.44909557172985;
pub const A_10_8: Scalar = 0.01059228297111661;
pub const A_10_9: Scalar = -0.0057468422638446166;
pub const A_11_1: Scalar = -1.7244701342624853;
pub const A_11_4: Scalar = -60.92349008483054;
pub const A_11_5: Scalar = -5.951518376222392;
pub const A_11_6: Scalar = 5.556523730698456;
pub const A_11_7: Scalar = 63.98301198033305;
pub const A_11_8: Scalar = 0.014642028250414961;
pub const A_11_9: Scalar = 0.06460408772358203;
pub const A_11_10: Scalar = -0.0793032316900888;
pub const A_12_1: Scalar = -3.301622667747079;
pub const A_12_4: Scalar = -118.01127235975251;
pub const A_12_5: Scalar = -10.141422388456112;
pub const A_12_6: Scalar = 9.139311332232058;
pub const A_12_7: Scalar = 123.37594282840426;
pub const A_12_8: Scalar = 4.62324437887458;
pub const A_12_9: Scalar = -3.3832777380682018;
pub const A_12_10: Scalar = 4.527592100324618;
pub const A_12_11: Scalar = -5.828495485811623;
pub const A_13_1: Scalar = -3.039515033766309;
pub const A_13_4: Scalar = -109.26086808941763;
pub const A_13_5: Scalar = -9.290642497400293;
pub const A_13_6: Scalar = 8.43050498176491;
pub const A_13_7: Scalar = 114.20100103783314;
pub const A_13_8: Scalar = -0.9637271342145479;
pub const A_13_9: Scalar = -5.0348840888021895;
pub const A_13_10: Scalar = 5.958130824002923;

pub const B_1: Scalar = 0.04427989419007951;
pub const B_6: Scalar = 0.3541049391724449;
pub const B_7: Scalar = 0.24796921549564377;
pub const B_8: Scalar = -15.694202038838085;
pub const B_9: Scalar = 25.084064965558564;
pub const B_10: Scalar = -31.738367786260277;
pub const B_11: Scalar = 22.938283273988784;
pub const B_12: Scalar = -0.2361324633071542;

pub const D_1: Scalar = -0.00003272103901028138;
pub const D_6: Scalar = -0.0005046250618777704;
pub const D_7: Scalar = 0.0001211723589784759;
pub const D_8: Scalar = -20.142336771313868;
pub const D_9: Scalar = 5.2371785994398286;
pub const D_10: Scalar = -8.156744408794658;
pub const D_11: Scalar = 22.938283273988784;
pub const D_12: Scalar = -0.2361324633071542;
pub const D_13: Scalar = 0.36016794372897754;

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Verner8 {
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

impl Default for Verner8 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 8.0,
            dt_cut: 0.5,
            dt_min: ABS_TOL,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for Verner8
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl VariableStep for Verner8 {
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

impl<Y, U> Explicit<Y, U> for Verner8
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 13;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U> VariableStepExplicit<Y, U> for Verner8
where
    Self: Explicit<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn error(dt: Scalar, k: &[Y]) -> Result<Scalar, String> {
        Ok(((&k[0] * D_1
            + &k[5] * D_6
            + &k[6] * D_7
            + &k[7] * D_8
            + &k[8] * D_9
            + &k[9] * D_10
            + &k[10] * D_11
            + &k[11] * D_12
            + &k[12] * D_13)
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
        *y_trial = &k[0] * (A_8_1 * dt)
            + &k[3] * (A_8_4 * dt)
            + &k[4] * (A_8_5 * dt)
            + &k[5] * (A_8_6 * dt)
            + &k[6] * (A_8_7 * dt)
            + y;
        k[7] = function(t + C_8 * dt, y_trial)?;
        *y_trial = &k[0] * (A_9_1 * dt)
            + &k[3] * (A_9_4 * dt)
            + &k[4] * (A_9_5 * dt)
            + &k[5] * (A_9_6 * dt)
            + &k[6] * (A_9_7 * dt)
            + &k[7] * (A_9_8 * dt)
            + y;
        k[8] = function(t + C_9 * dt, y_trial)?;
        *y_trial = &k[0] * (A_10_1 * dt)
            + &k[3] * (A_10_4 * dt)
            + &k[4] * (A_10_5 * dt)
            + &k[5] * (A_10_6 * dt)
            + &k[6] * (A_10_7 * dt)
            + &k[7] * (A_10_8 * dt)
            + &k[8] * (A_10_9 * dt)
            + y;
        k[9] = function(t + C_10 * dt, y_trial)?;
        *y_trial = &k[0] * (A_11_1 * dt)
            + &k[3] * (A_11_4 * dt)
            + &k[4] * (A_11_5 * dt)
            + &k[5] * (A_11_6 * dt)
            + &k[6] * (A_11_7 * dt)
            + &k[7] * (A_11_8 * dt)
            + &k[8] * (A_11_9 * dt)
            + &k[9] * (A_11_10 * dt)
            + y;
        k[10] = function(t + C_11 * dt, y_trial)?;
        *y_trial = &k[0] * (A_12_1 * dt)
            + &k[3] * (A_12_4 * dt)
            + &k[4] * (A_12_5 * dt)
            + &k[5] * (A_12_6 * dt)
            + &k[6] * (A_12_7 * dt)
            + &k[7] * (A_12_8 * dt)
            + &k[8] * (A_12_9 * dt)
            + &k[9] * (A_12_10 * dt)
            + &k[10] * (A_12_11 * dt)
            + y;
        k[11] = function(t + dt, y_trial)?;
        *y_trial = &k[0] * (A_13_1 * dt)
            + &k[3] * (A_13_4 * dt)
            + &k[4] * (A_13_5 * dt)
            + &k[5] * (A_13_6 * dt)
            + &k[6] * (A_13_7 * dt)
            + &k[7] * (A_13_8 * dt)
            + &k[8] * (A_13_9 * dt)
            + &k[9] * (A_13_10 * dt)
            + y;
        if k.len() == Self::SLOPES {
            k[12] = function(t + dt, y_trial)?;
        }
        *y_trial = (&k[0] * B_1
            + &k[5] * B_6
            + &k[6] * B_7
            + &k[7] * B_8
            + &k[8] * B_9
            + &k[9] * B_10
            + &k[10] * B_11
            + &k[11] * B_12)
            * dt
            + y;
        Ok(())
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Verner8
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
