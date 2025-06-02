#[cfg(test)]
mod test;

use super::{
    super::{Tensor, TensorRank0, TensorVec, Vector, interpolate::InterpolateSolution},
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_2: TensorRank0 = 0.05;
const C_3: TensorRank0 = 0.1065625;
const C_4: TensorRank0 = 0.15984375;
const C_5: TensorRank0 = 0.39;
const C_6: TensorRank0 = 0.465;
const C_7: TensorRank0 = 0.155;
const C_8: TensorRank0 = 0.943;
const C_9: TensorRank0 = 0.901802041735857;
const C_10: TensorRank0 = 0.909;
const C_11: TensorRank0 = 0.94;

const A_2_1: TensorRank0 = 0.05;
const A_3_1: TensorRank0 = -0.0069931640625;
const A_3_2: TensorRank0 = 0.1135556640625;
const A_4_1: TensorRank0 = 0.0399609375;
const A_4_3: TensorRank0 = 0.1198828125;
const A_5_1: TensorRank0 = 0.36139756280045754;
const A_5_3: TensorRank0 = -1.3415240667004928;
const A_5_4: TensorRank0 = 1.3701265039000352;
const A_6_1: TensorRank0 = 0.049047202797202795;
const A_6_4: TensorRank0 = 0.23509720422144048;
const A_6_5: TensorRank0 = 0.18085559298135673;
const A_7_1: TensorRank0 = 0.06169289044289044;
const A_7_4: TensorRank0 = 0.11236568314640277;
const A_7_5: TensorRank0 = -0.03885046071451367;
const A_7_6: TensorRank0 = 0.01979188712522046;
const A_8_1: TensorRank0 = -1.767630240222327;
const A_8_4: TensorRank0 = -62.5;
const A_8_5: TensorRank0 = -6.061889377376669;
const A_8_6: TensorRank0 = 5.6508231982227635;
const A_8_7: TensorRank0 = 65.62169641937624;
const A_9_1: TensorRank0 = -1.1809450665549708;
const A_9_4: TensorRank0 = -41.50473441114321;
const A_9_5: TensorRank0 = -4.434438319103725;
const A_9_6: TensorRank0 = 4.260408188586133;
const A_9_7: TensorRank0 = 43.75364022446172;
const A_9_8: TensorRank0 = 0.00787142548991231;
const A_10_1: TensorRank0 = -1.2814059994414884;
const A_10_4: TensorRank0 = -45.047139960139866;
const A_10_5: TensorRank0 = -4.731362069449576;
const A_10_6: TensorRank0 = 4.514967016593808;
const A_10_7: TensorRank0 = 47.44909557172985;
const A_10_8: TensorRank0 = 0.01059228297111661;
const A_10_9: TensorRank0 = -0.0057468422638446166;
const A_11_1: TensorRank0 = -1.7244701342624853;
const A_11_4: TensorRank0 = -60.92349008483054;
const A_11_5: TensorRank0 = -5.951518376222392;
const A_11_6: TensorRank0 = 5.556523730698456;
const A_11_7: TensorRank0 = 63.98301198033305;
const A_11_8: TensorRank0 = 0.014642028250414961;
const A_11_9: TensorRank0 = 0.06460408772358203;
const A_11_10: TensorRank0 = -0.0793032316900888;
const A_12_1: TensorRank0 = -3.301622667747079;
const A_12_4: TensorRank0 = -118.01127235975251;
const A_12_5: TensorRank0 = -10.141422388456112;
const A_12_6: TensorRank0 = 9.139311332232058;
const A_12_7: TensorRank0 = 123.37594282840426;
const A_12_8: TensorRank0 = 4.62324437887458;
const A_12_9: TensorRank0 = -3.3832777380682018;
const A_12_10: TensorRank0 = 4.527592100324618;
const A_12_11: TensorRank0 = -5.828495485811623;
const A_13_1: TensorRank0 = -3.039515033766309;
const A_13_4: TensorRank0 = -109.26086808941763;
const A_13_5: TensorRank0 = -9.290642497400293;
const A_13_6: TensorRank0 = 8.43050498176491;
const A_13_7: TensorRank0 = 114.20100103783314;
const A_13_8: TensorRank0 = -0.9637271342145479;
const A_13_9: TensorRank0 = -5.0348840888021895;
const A_13_10: TensorRank0 = 5.958130824002923;

const B_1: TensorRank0 = 0.04427989419007951;
const B_6: TensorRank0 = 0.3541049391724449;
const B_7: TensorRank0 = 0.24796921549564377;
const B_8: TensorRank0 = -15.694202038838085;
const B_9: TensorRank0 = 25.084064965558564;
const B_10: TensorRank0 = -31.738367786260277;
const B_11: TensorRank0 = 22.938283273988784;
const B_12: TensorRank0 = -0.2361324633071542;

const D_1: TensorRank0 = -0.00003272103901028138;
const D_6: TensorRank0 = -0.0005046250618777704;
const D_7: TensorRank0 = 0.0001211723589784759;
const D_8: TensorRank0 = -20.142336771313868;
const D_9: TensorRank0 = 5.2371785994398286;
const D_10: TensorRank0 = -8.156744408794658;
const D_11: TensorRank0 = 22.938283273988784;
const D_12: TensorRank0 = -0.2361324633071542;
const D_13: TensorRank0 = 0.36016794372897754;

/// Explicit, thirteen-stage, eighth-order, variable-step, Runge-Kutta method.[^cite]
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
pub struct Verner8 {
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

impl Default for Verner8 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 8.0,
            dt_init: 0.1,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Verner8
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
        time: &[TensorRank0],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = time[0];
        let mut dt = self.dt_init * time[time.len() - 1];
        let mut e;
        let mut k_1 = function(t, &initial_condition)?;
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
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut y = initial_condition.clone();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::zero(0);
        dydt_sol.push(k_1.clone());
        let mut y_trial;
        while t < time[time.len() - 1] {
            k_1 = function(t, &y)?;
            k_2 = function(t + C_2 * dt, &(&k_1 * (A_2_1 * dt) + &y))?;
            k_3 = function(
                t + C_3 * dt,
                &(&k_1 * (A_3_1 * dt) + &k_2 * (A_3_2 * dt) + &y),
            )?;
            k_4 = function(
                t + C_4 * dt,
                &(&k_1 * (A_4_1 * dt) + &k_3 * (A_4_3 * dt) + &y),
            )?;
            k_5 = function(
                t + C_5 * dt,
                &(&k_1 * (A_5_1 * dt) + &k_3 * (A_5_3 * dt) + &k_4 * (A_5_4 * dt) + &y),
            )?;
            k_6 = function(
                t + C_6 * dt,
                &(&k_1 * (A_6_1 * dt) + &k_4 * (A_6_4 * dt) + &k_5 * (A_6_5 * dt) + &y),
            )?;
            k_7 = function(
                t + C_7 * dt,
                &(&k_1 * (A_7_1 * dt)
                    + &k_4 * (A_7_4 * dt)
                    + &k_5 * (A_7_5 * dt)
                    + &k_6 * (A_7_6 * dt)
                    + &y),
            )?;
            k_8 = function(
                t + C_8 * dt,
                &(&k_1 * (A_8_1 * dt)
                    + &k_4 * (A_8_4 * dt)
                    + &k_5 * (A_8_5 * dt)
                    + &k_6 * (A_8_6 * dt)
                    + &k_7 * (A_8_7 * dt)
                    + &y),
            )?;
            k_9 = function(
                t + C_9 * dt,
                &(&k_1 * (A_9_1 * dt)
                    + &k_4 * (A_9_4 * dt)
                    + &k_5 * (A_9_5 * dt)
                    + &k_6 * (A_9_6 * dt)
                    + &k_7 * (A_9_7 * dt)
                    + &k_8 * (A_9_8 * dt)
                    + &y),
            )?;
            k_10 = function(
                t + C_10 * dt,
                &(&k_1 * (A_10_1 * dt)
                    + &k_4 * (A_10_4 * dt)
                    + &k_5 * (A_10_5 * dt)
                    + &k_6 * (A_10_6 * dt)
                    + &k_7 * (A_10_7 * dt)
                    + &k_8 * (A_10_8 * dt)
                    + &k_9 * (A_10_9 * dt)
                    + &y),
            )?;
            k_11 = function(
                t + C_11 * dt,
                &(&k_1 * (A_11_1 * dt)
                    + &k_4 * (A_11_4 * dt)
                    + &k_5 * (A_11_5 * dt)
                    + &k_6 * (A_11_6 * dt)
                    + &k_7 * (A_11_7 * dt)
                    + &k_8 * (A_11_8 * dt)
                    + &k_9 * (A_11_9 * dt)
                    + &k_10 * (A_11_10 * dt)
                    + &y),
            )?;
            k_12 = function(
                t + dt,
                &(&k_1 * (A_12_1 * dt)
                    + &k_4 * (A_12_4 * dt)
                    + &k_5 * (A_12_5 * dt)
                    + &k_6 * (A_12_6 * dt)
                    + &k_7 * (A_12_7 * dt)
                    + &k_8 * (A_12_8 * dt)
                    + &k_9 * (A_12_9 * dt)
                    + &k_10 * (A_12_10 * dt)
                    + &k_11 * (A_12_11 * dt)
                    + &y),
            )?;
            y_trial = (&k_1 * B_1
                + &k_6 * B_6
                + &k_7 * B_7
                + &k_8 * B_8
                + &k_9 * B_9
                + &k_10 * B_10
                + &k_11 * B_11
                + &k_12 * B_12)
                * dt
                + &y;
            k_13 = function(
                t + dt,
                &(&k_1 * (A_13_1 * dt)
                    + &k_4 * (A_13_4 * dt)
                    + &k_5 * (A_13_5 * dt)
                    + &k_6 * (A_13_6 * dt)
                    + &k_7 * (A_13_7 * dt)
                    + &k_8 * (A_13_8 * dt)
                    + &k_9 * (A_13_9 * dt)
                    + &k_10 * (A_13_10 * dt)
                    + &y),
            )?;
            e = ((&k_1 * D_1
                + &k_6 * D_6
                + &k_7 * D_7
                + &k_8 * D_8
                + &k_9 * D_9
                + &k_10 * D_10
                + &k_11 * D_11
                + &k_12 * D_12
                + &k_13 * D_13)
                * dt)
                .norm_inf();
            if e < self.abs_tol || e / y_trial.norm_inf() < self.rel_tol {
                t += dt;
                y = y_trial;
                t_sol.push(t);
                y_sol.push(y.clone());
                dydt_sol.push(function(t, &y)?);
            }
            dt *= self.dt_beta * (self.abs_tol / e).powf(1.0 / self.dt_expn);
        }
        if time.len() > 2 {
            let t_int = Vector::new(time);
            let (y_int, dydt_int) = self.interpolate(&t_int, &t_sol, &y_sol, function)?;
            Ok((t_int, y_int, dydt_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol))
        }
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Verner8
where
    Y: Tensor,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
    ) -> Result<(U, U), IntegrationError> {
        let mut dt;
        let mut i;
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
        let mut t;
        let mut y;
        let mut y_int = U::zero(0);
        let mut dydt_int = U::zero(0);
        let mut y_trial;
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i > time_k).unwrap();
            t = tp[i - 1];
            y = yp[i - 1].clone();
            dt = time_k - t;
            k_1 = function(t, &y)?;
            k_2 = function(t + C_2 * dt, &(&k_1 * (A_2_1 * dt) + &y))?;
            k_3 = function(
                t + C_3 * dt,
                &(&k_1 * (A_3_1 * dt) + &k_2 * (A_3_2 * dt) + &y),
            )?;
            k_4 = function(
                t + C_4 * dt,
                &(&k_1 * (A_4_1 * dt) + &k_3 * (A_4_3 * dt) + &y),
            )?;
            k_5 = function(
                t + C_5 * dt,
                &(&k_1 * (A_5_1 * dt) + &k_3 * (A_5_3 * dt) + &k_4 * (A_5_4 * dt) + &y),
            )?;
            k_6 = function(
                t + C_6 * dt,
                &(&k_1 * (A_6_1 * dt) + &k_4 * (A_6_4 * dt) + &k_5 * (A_6_5 * dt) + &y),
            )?;
            k_7 = function(
                t + C_7 * dt,
                &(&k_1 * (A_7_1 * dt)
                    + &k_4 * (A_7_4 * dt)
                    + &k_5 * (A_7_5 * dt)
                    + &k_6 * (A_7_6 * dt)
                    + &y),
            )?;
            k_8 = function(
                t + C_8 * dt,
                &(&k_1 * (A_8_1 * dt)
                    + &k_4 * (A_8_4 * dt)
                    + &k_5 * (A_8_5 * dt)
                    + &k_6 * (A_8_6 * dt)
                    + &k_7 * (A_8_7 * dt)
                    + &y),
            )?;
            k_9 = function(
                t + C_9 * dt,
                &(&k_1 * (A_9_1 * dt)
                    + &k_4 * (A_9_4 * dt)
                    + &k_5 * (A_9_5 * dt)
                    + &k_6 * (A_9_6 * dt)
                    + &k_7 * (A_9_7 * dt)
                    + &k_8 * (A_9_8 * dt)
                    + &y),
            )?;
            k_10 = function(
                t + C_10 * dt,
                &(&k_1 * (A_10_1 * dt)
                    + &k_4 * (A_10_4 * dt)
                    + &k_5 * (A_10_5 * dt)
                    + &k_6 * (A_10_6 * dt)
                    + &k_7 * (A_10_7 * dt)
                    + &k_8 * (A_10_8 * dt)
                    + &k_9 * (A_10_9 * dt)
                    + &y),
            )?;
            k_11 = function(
                t + C_11 * dt,
                &(&k_1 * (A_11_1 * dt)
                    + &k_4 * (A_11_4 * dt)
                    + &k_5 * (A_11_5 * dt)
                    + &k_6 * (A_11_6 * dt)
                    + &k_7 * (A_11_7 * dt)
                    + &k_8 * (A_11_8 * dt)
                    + &k_9 * (A_11_9 * dt)
                    + &k_10 * (A_11_10 * dt)
                    + &y),
            )?;
            k_12 = function(
                t + dt,
                &(&k_1 * (A_12_1 * dt)
                    + &k_4 * (A_12_4 * dt)
                    + &k_5 * (A_12_5 * dt)
                    + &k_6 * (A_12_6 * dt)
                    + &k_7 * (A_12_7 * dt)
                    + &k_8 * (A_12_8 * dt)
                    + &k_9 * (A_12_9 * dt)
                    + &k_10 * (A_12_10 * dt)
                    + &k_11 * (A_12_11 * dt)
                    + &y),
            )?;
            y_trial = (&k_1 * B_1
                + &k_6 * B_6
                + &k_7 * B_7
                + &k_8 * B_8
                + &k_9 * B_9
                + &k_10 * B_10
                + &k_11 * B_11
                + &k_12 * B_12)
                * dt
                + &y;
            dydt_int.push(function(t + dt, &y_trial)?);
            y_int.push(y_trial);
        }
        Ok((y_int, dydt_int))
    }
}
