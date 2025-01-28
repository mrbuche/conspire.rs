#[cfg(test)]
mod test;

use super::{
    super::{
        interpolate::InterpolateSolution, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
    },
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_44_45: TensorRank0 = 44.0 / 45.0;
const C_56_15: TensorRank0 = 56.0 / 15.0;
const C_32_9: TensorRank0 = 32.0 / 9.0;
const C_8_9: TensorRank0 = 8.0 / 9.0;
const C_19372_6561: TensorRank0 = 19372.0 / 6561.0;
const C_25360_2187: TensorRank0 = 25360.0 / 2187.0;
const C_64448_6561: TensorRank0 = 64448.0 / 6561.0;
const C_212_729: TensorRank0 = 212.0 / 729.0;
const C_9017_3168: TensorRank0 = 9017.0 / 3168.0;
const C_355_33: TensorRank0 = 355.0 / 33.0;
const C_46732_5247: TensorRank0 = 46732.0 / 5247.0;
const C_49_176: TensorRank0 = 49.0 / 176.0;
const C_5103_18656: TensorRank0 = 5103.0 / 18656.0;
const C_35_384: TensorRank0 = 35.0 / 384.0;
const C_500_1113: TensorRank0 = 500.0 / 1113.0;
const C_125_192: TensorRank0 = 125.0 / 192.0;
const C_2187_6784: TensorRank0 = 2187.0 / 6784.0;
const C_11_84: TensorRank0 = 11.0 / 84.0;
const C_71_57600: TensorRank0 = 71.0 / 57600.0;
const C_71_16695: TensorRank0 = 71.0 / 16695.0;
const C_71_1920: TensorRank0 = 71.0 / 1920.0;
const C_17253_339200: TensorRank0 = 17253.0 / 339200.0;
const C_22_525: TensorRank0 = 22.0 / 525.0;

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
#[derive(Debug)]
pub struct Ode45 {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Multiplying factor when decreasing time steps.
    pub dec_fac: TensorRank0,
    /// Initial relative timestep.
    pub dt_init: TensorRank0,
    /// Multiplying factor when increasing time steps.
    pub inc_fac: TensorRank0,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
}

impl Default for Ode45 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dec_fac: 0.5,
            dt_init: 0.1,
            inc_fac: 1.1,
            rel_tol: REL_TOL,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Ode45
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        initial_time: TensorRank0,
        initial_condition: Y,
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut dt = self.dt_init * time[time.len() - 1];
        let mut e;
        let mut k_1 = function(&initial_time, &initial_condition);
        let mut k_2;
        let mut k_3;
        let mut k_4;
        let mut k_5;
        let mut k_6;
        let mut k_7;
        let mut t = time[0];
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut y = initial_condition.copy();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.copy());
        let mut y_trial;
        while t < time[time.len() - 1] {
            k_2 = function(&(t + 0.2 * dt), &(&k_1 * (0.2 * dt) + &y));
            k_3 = function(
                &(t + 0.3 * dt),
                &(&k_1 * (0.075 * dt) + &k_2 * (0.225 * dt) + &y),
            );
            k_4 = function(
                &(t + 0.8 * dt),
                &(&k_1 * (C_44_45 * dt) - &k_2 * (C_56_15 * dt) + &k_3 * (C_32_9 * dt) + &y),
            );
            k_5 = function(
                &(t + C_8_9 * dt),
                &(&k_1 * (C_19372_6561 * dt) - &k_2 * (C_25360_2187 * dt)
                    + &k_3 * (C_64448_6561 * dt)
                    - &k_4 * (C_212_729 * dt)
                    + &y),
            );
            k_6 = function(
                &(t + dt),
                &(&k_1 * (C_9017_3168 * dt) - &k_2 * (C_355_33 * dt)
                    + &k_3 * (C_46732_5247 * dt)
                    + &k_4 * (C_49_176 * dt)
                    - &k_5 * (C_5103_18656 * dt)
                    + &y),
            );
            y_trial = (&k_1 * C_35_384 + &k_3 * C_500_1113 + &k_4 * C_125_192 - &k_5 * C_2187_6784
                + &k_6 * C_11_84)
                * dt
                + &y;
            k_7 = function(&(t + dt), &y_trial);
            e = ((&k_1 * C_71_57600 - k_3 * C_71_16695 + k_4 * C_71_1920 - k_5 * C_17253_339200
                + k_6 * C_22_525
                - &k_7 * 0.025)
                * dt)
                .norm();
            if e < self.abs_tol || e / y_trial.norm() < self.rel_tol {
                k_1 = k_7;
                t += dt;
                dt *= self.inc_fac;
                y = y_trial;
                t_sol.push(t.copy());
                y_sol.push(y.copy());
            } else {
                dt *= self.dec_fac;
            }
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

impl<Y, U> InterpolateSolution<Y, U> for Ode45
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        ti: &Vector,
        tp: &Vector,
        yp: &U,
        f: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        let mut dt = 0.0;
        let mut i = 0;
        let mut k_1 = Y::zero();
        let mut k_2 = Y::zero();
        let mut k_3 = Y::zero();
        let mut k_4 = Y::zero();
        let mut k_5 = Y::zero();
        let mut k_6 = Y::zero();
        let mut t = 0.0;
        let mut y = Y::zero();
        ti.iter()
            .map(|ti_k| {
                i = tp.iter().position(|tp_i| tp_i > ti_k).unwrap();
                t = tp[i - 1].copy();
                y = yp[i - 1].copy();
                dt = ti_k - t;
                k_1 = f(&t, &y);
                k_2 = f(&(t + 0.2 * dt), &(&k_1 * (0.2 * dt) + &y));
                k_3 = f(
                    &(t + 0.3 * dt),
                    &(&k_1 * (0.075 * dt) + &k_2 * (0.225 * dt) + &y),
                );
                k_4 = f(
                    &(t + 0.8 * dt),
                    &(&k_1 * (C_44_45 * dt) - &k_2 * (C_56_15 * dt) + &k_3 * (C_32_9 * dt) + &y),
                );
                k_5 = f(
                    &(t + C_8_9 * dt),
                    &(&k_1 * (C_19372_6561 * dt) - &k_2 * (C_25360_2187 * dt)
                        + &k_3 * (C_64448_6561 * dt)
                        - &k_4 * (C_212_729 * dt)
                        + &y),
                );
                k_6 = f(
                    &(t + dt),
                    &(&k_1 * (C_9017_3168 * dt) - &k_2 * (C_355_33 * dt)
                        + &k_3 * (C_46732_5247 * dt)
                        + &k_4 * (C_49_176 * dt)
                        - &k_5 * (C_5103_18656 * dt)
                        + &y),
                );
                (&k_1 * C_35_384 + &k_3 * C_500_1113 + &k_4 * C_125_192 - &k_5 * C_2187_6784
                    + &k_6 * C_11_84)
                    * dt
                    + &y
            })
            .collect()
    }
}
