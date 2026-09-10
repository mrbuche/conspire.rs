#[cfg(test)]
mod test;

use crate::math::Norm;
use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        ButcherTableau, EmbeddedTableau, Explicit, IntegrationError, OdeIntegrator, Times,
        VariableStep, VariableStepExplicit, VariableStepExplicitFirstSameAsLast,
    },
    interpolate::InterpolateSolution,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

pub(crate) const P_1_0: Scalar = 1.0;
pub(crate) const P_1_1: Scalar = -8048581381.0 / 2820520608.0;
pub(crate) const P_1_2: Scalar = 8663915743.0 / 2820520608.0;
pub(crate) const P_1_3: Scalar = -12715105075.0 / 11282082432.0;
pub(crate) const P_3_1: Scalar = 131558114200.0 / 32700410799.0;
pub(crate) const P_3_2: Scalar = -68118460800.0 / 10900136933.0;
pub(crate) const P_3_3: Scalar = 87487479700.0 / 32700410799.0;
pub(crate) const P_4_1: Scalar = -1754552775.0 / 470086768.0;
pub(crate) const P_4_2: Scalar = 14199869525.0 / 1410260304.0;
pub(crate) const P_4_3: Scalar = -10690763975.0 / 1880347072.0;
pub(crate) const P_5_1: Scalar = 127303824393.0 / 49829197408.0;
pub(crate) const P_5_2: Scalar = -318862633887.0 / 49829197408.0;
pub(crate) const P_5_3: Scalar = 701980252875.0 / 199316789632.0;
pub(crate) const P_6_1: Scalar = -282668133.0 / 205662961.0;
pub(crate) const P_6_2: Scalar = 2019193451.0 / 616988883.0;
pub(crate) const P_6_3: Scalar = -1453857185.0 / 822651844.0;
pub(crate) const P_7_1: Scalar = 40617522.0 / 29380423.0;
pub(crate) const P_7_2: Scalar = -110615467.0 / 29380423.0;
pub(crate) const P_7_3: Scalar = 69997945.0 / 29380423.0;

/// The Dormand–Prince 5(4) tableau.
#[derive(Debug)]
pub struct Tableau;

impl ButcherTableau for Tableau {
    const STAGES: usize = 7;
    const ORDER: Scalar = 5.0;
    const A: &'static [&'static [Scalar]] = &[
        &[],
        &[0.2],
        &[0.075, 0.225],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        &[
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        &[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];
    const C: &'static [Scalar] = &[0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];
    const B: &'static [Scalar] = &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];
    const FSAL: bool = true;
}

impl EmbeddedTableau for Tableau {
    const D: &'static [Scalar] = &[
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -0.025,
    ];
}

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
    pub error_norm: Norm,
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
            error_norm: Norm::Chebyshev,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for DormandPrince
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> VariableStep<T> for DormandPrince {
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
    fn dt_min(&self) -> Quantity<T> {
        Quantity::new(self.dt_min)
    }
    fn error_norm(&self) -> &Norm {
        &self.error_norm
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for DormandPrince
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 7;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> VariableStepExplicit<Y, U, V, T> for DormandPrince
where
    Self: Explicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    type Tableau = Tableau;
    fn slopes_and_error(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        self.slopes_and_error_fsal(function, y, t, dt, k, y_trial)
    }
    fn step(
        &self,
        _function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &mut Y,
        t: &mut Quantity<T>,
        y_sol: &mut U,
        t_sol: &mut Times<T>,
        dydt_sol: &mut V,
        k_sol: &mut Vec<V>,
        dt: &mut Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        self.step_fsal(y, t, y_sol, t_sol, dydt_sol, k_sol, dt, k, y_trial, e)
    }
}

impl<Y, U, V, T> VariableStepExplicitFirstSameAsLast<Y, U, V, T> for DormandPrince
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
}

impl DormandPrince {
    pub(crate) fn interpolate_free_dense<Y, U, V, T>(
        time: &Times<T>,
        tp: &Times<T>,
        yp: &U,
        dydtp: &V,
        k_sol: &[V],
    ) -> (U, V)
    where
        Y: Differentiate<T> + Tensor,
        Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
        for<'a> &'a Derivative<Y, T>:
            Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
        U: TensorVec<Item = Y>,
        V: TensorVec<Item = Derivative<Y, T>>,
    {
        let mut y_int = U::new();
        let mut dydt_int = V::new();
        for time_k in time.iter() {
            let i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                y_int.push(yp[i].clone());
                dydt_int.push(dydtp[i].clone());
            } else {
                let t_0 = tp[i - 1];
                let h = tp[i] - t_0;
                let theta = (*time_k - t_0).value() / h.value();
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
        (y_int, dydt_int)
    }
}

impl<Y, U, V, T> InterpolateSolution<Y, U, V, T> for DormandPrince
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn interpolate(
        &self,
        time: &Times<T>,
        tp: &Times<T>,
        yp: &U,
        dydtp: &V,
        k_sol: &[V],
        _function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
    ) -> Result<(U, V), IntegrationError> {
        Ok(Self::interpolate_free_dense(time, tp, yp, dydtp, k_sol))
    }
}
