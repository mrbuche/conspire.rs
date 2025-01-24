#[cfg(test)]
mod test;

use super::{
    super::{
        interpolate::{Interpolate1D, LinearInterpolation},
        Tensor, TensorRank0, TensorVec, Vector,
    },
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Index, Mul, Sub};

/// Explicit, three-stage, third-order, variable-step, Runge-Kutta method.[^cite]
///
/// [^cite]: P. Bogacki and L.F. Shampine, [Appl. Math. Lett. **2**, 321 (1989)](https://doi.org/10.1016/0893-9659(89)90079-7).
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
/// k_2 = f(t_n + \tfrac{1}{2} h, y_n + \tfrac{1}{2} h k_1)
/// ```
/// ```math
/// k_3 = f(t_n + \tfrac{3}{4} h, y_n + \tfrac{3}{4} h k_2)
/// ```
/// ```math
/// y_{n+1} = y_n + \frac{h}{9}\left(2k_1 + 3k_2 + 4k_3\right)
/// ```
/// ```math
/// k_4 = f(t_{n+1}, y_{n+1})
/// ```
/// ```math
/// e_{n+1} = \frac{h}{72}\left(-5k_1 + 6k_2 + 8k_3 - 9k_4\right)
/// ```
#[derive(Debug)]
pub struct Ode23 {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Multiplying factor when decreasing time steps.
    pub dec_fac: TensorRank0,
    /// Multiplying factor when increasing time steps.
    pub inc_fac: TensorRank0,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
}

impl Default for Ode23 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dec_fac: 0.5,
            inc_fac: 1.1,
            rel_tol: REL_TOL,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Ode23
where
    Y: Tensor,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: FromIterator<Y> + Index<usize, Output = Y> + Tensor<Item = Y> + TensorVec<Item = Y>,
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
        let mut dt = 1e-1 * time[time.len() - 1];
        let mut e;
        let mut k_1 = function(&initial_time, &initial_condition);
        let mut k_2;
        let mut k_3;
        let mut k_4;
        let mut t = time[0];
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut y = initial_condition.copy();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.copy());
        let mut y_trial;
        while t < time[time.len() - 1] {
            k_2 = function(&(t + 0.5 * dt), &(&k_1 * (0.5 * dt) + &y));
            k_3 = function(&(t + 0.75 * dt), &(&k_2 * (0.75 * dt) + &y));
            y_trial = (&k_1 * 2.0 + &k_2 * 3.0 + &k_3 * 4.0) * (dt / 9.0) + &y;
            k_4 = function(&(t + dt), &y_trial);
            e = ((&k_1 * -5.0 + k_2 * 6.0 + k_3 * 8.0 + &k_4 * -9.0) * (dt / 72.0)).norm();
            if e < self.abs_tol || e / y_trial.norm() < self.rel_tol {
                k_1 = k_4;
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
            let y_int = LinearInterpolation::interpolate_1d(&t_int, &t_sol, &y_sol);
            Ok((t_int, y_int))
        } else {
            Ok((t_sol, y_sol))
        }
    }
}
