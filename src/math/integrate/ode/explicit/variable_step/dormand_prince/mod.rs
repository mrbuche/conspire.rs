#[cfg(test)]
mod test;

use crate::math::Norm;
use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        Explicit, IntegrationError, OdeIntegrator, VariableStep, VariableStepExplicit,
        VariableStepExplicitFirstSameAsLast,
    },
    interpolate::InterpolateSolution,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

pub const C_44_45: Scalar = 44.0 / 45.0;
pub const C_56_15: Scalar = 56.0 / 15.0;
pub const C_32_9: Scalar = 32.0 / 9.0;
pub const C_8_9: Scalar = 8.0 / 9.0;
pub const C_19372_6561: Scalar = 19372.0 / 6561.0;
pub const C_25360_2187: Scalar = 25360.0 / 2187.0;
pub const C_64448_6561: Scalar = 64448.0 / 6561.0;
pub const C_212_729: Scalar = 212.0 / 729.0;
pub const C_9017_3168: Scalar = 9017.0 / 3168.0;
pub const C_355_33: Scalar = 355.0 / 33.0;
pub const C_46732_5247: Scalar = 46732.0 / 5247.0;
pub const C_49_176: Scalar = 49.0 / 176.0;
pub const C_5103_18656: Scalar = 5103.0 / 18656.0;
pub const C_35_384: Scalar = 35.0 / 384.0;
pub const C_500_1113: Scalar = 500.0 / 1113.0;
pub const C_125_192: Scalar = 125.0 / 192.0;
pub const C_2187_6784: Scalar = 2187.0 / 6784.0;
pub const C_11_84: Scalar = 11.0 / 84.0;
pub const C_71_57600: Scalar = 71.0 / 57600.0;
pub const C_71_16695: Scalar = 71.0 / 16695.0;
pub const C_71_1920: Scalar = 71.0 / 1920.0;
pub const C_17253_339200: Scalar = 17253.0 / 339200.0;
pub const C_22_525: Scalar = 22.0 / 525.0;

pub const P_1_0: Scalar = 1.0;
pub const P_1_1: Scalar = -8048581381.0 / 2820520608.0;
pub const P_1_2: Scalar = 8663915743.0 / 2820520608.0;
pub const P_1_3: Scalar = -12715105075.0 / 11282082432.0;
pub const P_3_1: Scalar = 131558114200.0 / 32700410799.0;
pub const P_3_2: Scalar = -68118460800.0 / 10900136933.0;
pub const P_3_3: Scalar = 87487479700.0 / 32700410799.0;
pub const P_4_1: Scalar = -1754552775.0 / 470086768.0;
pub const P_4_2: Scalar = 14199869525.0 / 1410260304.0;
pub const P_4_3: Scalar = -10690763975.0 / 1880347072.0;
pub const P_5_1: Scalar = 127303824393.0 / 49829197408.0;
pub const P_5_2: Scalar = -318862633887.0 / 49829197408.0;
pub const P_5_3: Scalar = 701980252875.0 / 199316789632.0;
pub const P_6_1: Scalar = -282668133.0 / 205662961.0;
pub const P_6_2: Scalar = 2019193451.0 / 616988883.0;
pub const P_6_3: Scalar = -1453857185.0 / 822651844.0;
pub const P_7_1: Scalar = 40617522.0 / 29380423.0;
pub const P_7_2: Scalar = -110615467.0 / 29380423.0;
pub const P_7_3: Scalar = 69997945.0 / 29380423.0;

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct DormandPrince {
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
    /// Norm type for error evaluation.
    pub norm: Norm,
}

impl Default for DormandPrince {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 5.0,
            dt_cut: 0.5,
            dt_min: ABS_TOL,
            norm: Norm::Chebyshev,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for DormandPrince
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl VariableStep for DormandPrince {
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
    fn norm(&self) -> &Norm {
        &self.norm
    }
}

impl<Y, U> Explicit<Y, U> for DormandPrince
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 7;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U> VariableStepExplicit<Y, U> for DormandPrince
where
    Self: Explicit<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn error(&self, dt: Scalar, k: &[Y]) -> Result<Scalar, String> {
        Ok(self.norm.apply(
            &((&k[0] * C_71_57600 - &k[2] * C_71_16695 + &k[3] * C_71_1920
                - &k[4] * C_17253_339200
                + &k[5] * C_22_525
                - &k[6] * 0.025)
                * dt),
        ))
    }
    fn slopes(
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        *y_trial = &k[0] * (0.2 * dt) + y;
        k[1] = function(t + 0.2 * dt, y_trial)?;
        *y_trial = &k[0] * (0.075 * dt) + &k[1] * (0.225 * dt) + y;
        k[2] = function(t + 0.3 * dt, y_trial)?;
        *y_trial = &k[0] * (C_44_45 * dt) - &k[1] * (C_56_15 * dt) + &k[2] * (C_32_9 * dt) + y;
        k[3] = function(t + 0.8 * dt, y_trial)?;
        *y_trial = &k[0] * (C_19372_6561 * dt) - &k[1] * (C_25360_2187 * dt)
            + &k[2] * (C_64448_6561 * dt)
            - &k[3] * (C_212_729 * dt)
            + y;
        k[4] = function(t + C_8_9 * dt, y_trial)?;
        *y_trial = &k[0] * (C_9017_3168 * dt) - &k[1] * (C_355_33 * dt)
            + &k[2] * (C_46732_5247 * dt)
            + &k[3] * (C_49_176 * dt)
            - &k[4] * (C_5103_18656 * dt)
            + y;
        k[5] = function(t + dt, y_trial)?;
        *y_trial = (&k[0] * C_35_384 + &k[2] * C_500_1113 + &k[3] * C_125_192
            - &k[4] * C_2187_6784
            + &k[5] * C_11_84)
            * dt
            + y;
        Ok(())
    }
    fn slopes_and_error(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        self.slopes_and_error_fsal(function, y, t, dt, k, y_trial)
    }
    fn step(
        &self,
        _function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        k_sol: &mut Vec<U>,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        self.step_fsal(y, t, y_sol, t_sol, dydt_sol, k_sol, dt, k, y_trial, e)
    }
}

impl<Y, U> VariableStepExplicitFirstSameAsLast<Y, U> for DormandPrince
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
}

impl<Y, U> InterpolateSolution<Y, U> for DormandPrince
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
        dydtp: &U,
        k_sol: &[U],
        _function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        let mut y_int = U::new();
        let mut dydt_int = U::new();
        for time_k in time.iter() {
            let i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                y_int.push(yp[i].clone());
                dydt_int.push(dydtp[i].clone());
            } else {
                let t_0 = tp[i - 1];
                let h = tp[i] - t_0;
                let theta = (time_k - t_0) / h;
                let theta2 = theta * theta;
                let theta3 = theta2 * theta;
                let theta4 = theta3 * theta;
                let k = &k_sol[i - 1];
                let c_1 = theta * P_1_0 + theta2 * P_1_1 + theta3 * P_1_2 + theta4 * P_1_3;
                let c_3 = theta2 * P_3_1 + theta3 * P_3_2 + theta4 * P_3_3;
                let c_4 = theta2 * P_4_1 + theta3 * P_4_2 + theta4 * P_4_3;
                let c_5 = theta2 * P_5_1 + theta3 * P_5_2 + theta4 * P_5_3;
                let c_6 = theta2 * P_6_1 + theta3 * P_6_2 + theta4 * P_6_3;
                let c_7 = theta2 * P_7_1 + theta3 * P_7_2 + theta4 * P_7_3;
                let dc_1 =
                    P_1_0 + 2.0 * theta * P_1_1 + 3.0 * theta2 * P_1_2 + 4.0 * theta3 * P_1_3;
                let dc_3 = 2.0 * theta * P_3_1 + 3.0 * theta2 * P_3_2 + 4.0 * theta3 * P_3_3;
                let dc_4 = 2.0 * theta * P_4_1 + 3.0 * theta2 * P_4_2 + 4.0 * theta3 * P_4_3;
                let dc_5 = 2.0 * theta * P_5_1 + 3.0 * theta2 * P_5_2 + 4.0 * theta3 * P_5_3;
                let dc_6 = 2.0 * theta * P_6_1 + 3.0 * theta2 * P_6_2 + 4.0 * theta3 * P_6_3;
                let dc_7 = 2.0 * theta * P_7_1 + 3.0 * theta2 * P_7_2 + 4.0 * theta3 * P_7_3;
                let sum = &k[0] * c_1
                    + &k[2] * c_3
                    + &k[3] * c_4
                    + &k[4] * c_5
                    + &k[5] * c_6
                    + &k[6] * c_7;
                y_int.push(sum * h + &yp[i - 1]);
                dydt_int.push(
                    &k[0] * dc_1
                        + &k[2] * dc_3
                        + &k[3] * dc_4
                        + &k[4] * dc_5
                        + &k[5] * dc_6
                        + &k[6] * dc_7,
                );
            }
        }
        Ok((y_int, dydt_int))
    }
}
