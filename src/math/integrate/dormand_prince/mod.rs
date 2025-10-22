#[cfg(test)]
mod test;

use super::{
    super::{Scalar, Tensor, TensorVec, Vector, interpolate::InterpolateSolution},
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_44_45: Scalar = 44.0 / 45.0;
const C_56_15: Scalar = 56.0 / 15.0;
const C_32_9: Scalar = 32.0 / 9.0;
const C_8_9: Scalar = 8.0 / 9.0;
const C_19372_6561: Scalar = 19372.0 / 6561.0;
const C_25360_2187: Scalar = 25360.0 / 2187.0;
const C_64448_6561: Scalar = 64448.0 / 6561.0;
const C_212_729: Scalar = 212.0 / 729.0;
const C_9017_3168: Scalar = 9017.0 / 3168.0;
const C_355_33: Scalar = 355.0 / 33.0;
const C_46732_5247: Scalar = 46732.0 / 5247.0;
const C_49_176: Scalar = 49.0 / 176.0;
const C_5103_18656: Scalar = 5103.0 / 18656.0;
const C_35_384: Scalar = 35.0 / 384.0;
const C_500_1113: Scalar = 500.0 / 1113.0;
const C_125_192: Scalar = 125.0 / 192.0;
const C_2187_6784: Scalar = 2187.0 / 6784.0;
const C_11_84: Scalar = 11.0 / 84.0;
const C_71_57600: Scalar = 71.0 / 57600.0;
const C_71_16695: Scalar = 71.0 / 16695.0;
const C_71_1920: Scalar = 71.0 / 1920.0;
const C_17253_339200: Scalar = 17253.0 / 339200.0;
const C_22_525: Scalar = 22.0 / 525.0;

/// Explicit, six-stage, fifth-order, variable-step, Runge-Kutta method.[^cite]
///
/// [^cite]: J.R. Dormand and P.J. Prince, [J. Comput. Appl. Math. **6**, 19 (1980)](https://doi.org/10.1016/0771-050X(80)90013-3).
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
/// k_2 = f(t_n + \tfrac{1}{5} h, y_n + \tfrac{1}{5} h k_1)
/// ```
/// ```math
/// k_3 = f(t_n + \tfrac{3}{10} h, y_n + \tfrac{3}{40} h k_1 + \tfrac{9}{40} h k_2)
/// ```
/// ```math
/// k_4 = f(t_n + \tfrac{4}{5} h, y_n + \tfrac{44}{45} h k_1 - \tfrac{56}{15} h k_2 + \tfrac{32}{9} h k_3)
/// ```
/// ```math
/// k_5 = f(t_n + \tfrac{8}{9} h, y_n + \tfrac{19372}{6561} h k_1 - \tfrac{25360}{2187} h k_2 + \tfrac{64448}{6561} h k_3 - \tfrac{212}{729} h k_4)
/// ```
/// ```math
/// k_6 = f(t_n + h, y_n + \tfrac{9017}{3168} h k_1 - \tfrac{355}{33} h k_2 - \tfrac{46732}{5247} h k_3 + \tfrac{49}{176} h k_4 - \tfrac{5103}{18656} h k_5)
/// ```
/// ```math
/// y_{n+1} = y_n + h\left(\frac{35}{384}\,k_1 + \frac{500}{1113}\,k_3 + \frac{125}{192}\,k_4 - \frac{2187}{6784}\,k_5 + \frac{11}{84}\,k_6\right)
/// ```
/// ```math
/// k_7 = f(t_{n+1}, y_{n+1})
/// ```
/// ```math
/// e_{n+1} = \frac{h}{5}\left(\frac{71}{11520}\,k_1 - \frac{71}{3339}\,k_3 + \frac{71}{384}\,k_4 - \frac{17253}{67840}\,k_5 + \frac{22}{105}\,k_6 - \frac{1}{8}\,k_7\right)
/// ```
/// ```math
/// h_{n+1} = \beta h \left(\frac{e_\mathrm{tol}}{e_{n+1}}\right)^{1/p}
/// ```
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
}

impl Default for DormandPrince {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 5.0,
        }
    }
}

impl<Y, U> Explicit<Y, U> for DormandPrince
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 7;
    fn slopes(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: &Scalar,
        dt: &Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        k[1] = function(t + 0.2 * dt, &(&k[0] * (0.2 * dt) + y))?;
        k[2] = function(
            t + 0.3 * dt,
            &(&k[0] * (0.075 * dt) + &k[1] * (0.225 * dt) + y),
        )?;
        k[3] = function(
            t + 0.8 * dt,
            &(&k[0] * (C_44_45 * dt) - &k[1] * (C_56_15 * dt) + &k[2] * (C_32_9 * dt) + y),
        )?;
        k[4] = function(
            t + C_8_9 * dt,
            &(&k[0] * (C_19372_6561 * dt) - &k[1] * (C_25360_2187 * dt)
                + &k[2] * (C_64448_6561 * dt)
                - &k[3] * (C_212_729 * dt)
                + y),
        )?;
        k[5] = function(
            t + dt,
            &(&k[0] * (C_9017_3168 * dt) - &k[1] * (C_355_33 * dt)
                + &k[2] * (C_46732_5247 * dt)
                + &k[3] * (C_49_176 * dt)
                - &k[4] * (C_5103_18656 * dt)
                + y),
        )?;
        *y_trial = (&k[0] * C_35_384 + &k[2] * C_500_1113 + &k[3] * C_125_192
            - &k[4] * C_2187_6784
            + &k[5] * C_11_84)
            * *dt
            + y;
        k[6] = function(t + dt, y_trial)?;
        Ok(
            ((&k[0] * C_71_57600 - &k[2] * C_71_16695 + &k[3] * C_71_1920
                - &k[4] * C_17253_339200
                + &k[5] * C_22_525
                - &k[6] * 0.025)
                * *dt)
                .norm_inf(),
        )
    }
    fn step(
        &self,
        _function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        e: &Scalar,
    ) -> Result<(), String> {
        if e < &self.abs_tol || e / y_trial.norm_inf() < self.rel_tol {
            k[0] = k[6].clone();
            *t += *dt;
            *y = y_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            dydt_sol.push(k[0].clone());
        }
        if e > &0.0 {
            *dt *= self.dt_beta * (self.abs_tol / e).powf(1.0 / self.dt_expn)
        }
        Ok(())
    }
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
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k_1;
        let mut k_2;
        let mut k_3;
        let mut k_4;
        let mut k_5;
        let mut k_6;
        let mut t;
        let mut y;
        let mut y_int = U::new();
        let mut dydt_int = U::new();
        let mut y_trial;
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                dt = 0.0;
            } else {
                t = tp[i - 1];
                y = yp[i - 1].clone();
                dt = time_k - t;
                k_1 = function(t, &y)?;
                k_2 = function(t + 0.2 * dt, &(&k_1 * (0.2 * dt) + &y))?;
                k_3 = function(
                    t + 0.3 * dt,
                    &(&k_1 * (0.075 * dt) + &k_2 * (0.225 * dt) + &y),
                )?;
                k_4 = function(
                    t + 0.8 * dt,
                    &(&k_1 * (C_44_45 * dt) - &k_2 * (C_56_15 * dt) + &k_3 * (C_32_9 * dt) + &y),
                )?;
                k_5 = function(
                    t + C_8_9 * dt,
                    &(&k_1 * (C_19372_6561 * dt) - &k_2 * (C_25360_2187 * dt)
                        + &k_3 * (C_64448_6561 * dt)
                        - &k_4 * (C_212_729 * dt)
                        + &y),
                )?;
                k_6 = function(
                    t + dt,
                    &(&k_1 * (C_9017_3168 * dt) - &k_2 * (C_355_33 * dt)
                        + &k_3 * (C_46732_5247 * dt)
                        + &k_4 * (C_49_176 * dt)
                        - &k_5 * (C_5103_18656 * dt)
                        + &y),
                )?;
                y_trial = (&k_1 * C_35_384 + &k_3 * C_500_1113 + &k_4 * C_125_192
                    - &k_5 * C_2187_6784
                    + &k_6 * C_11_84)
                    * dt
                    + &y;
            }
            dydt_int.push(function(t + dt, &y_trial)?);
            y_int.push(y_trial);
        }
        Ok((y_int, dydt_int))
    }
}
