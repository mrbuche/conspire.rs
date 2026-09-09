#[cfg(test)]
mod test;

use crate::math::Norm;
use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        ButcherTableau, EmbeddedTableau, Explicit, IntegrationError, OdeIntegrator, Times,
        VariableStep, VariableStepExplicit,
    },
    interpolate::InterpolateSolution,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

pub(crate) const C_2: Scalar = 0.05;
pub(crate) const C_3: Scalar = 0.1065625;
pub(crate) const C_4: Scalar = 0.15984375;
pub(crate) const C_5: Scalar = 0.39;
pub(crate) const C_6: Scalar = 0.465;
pub(crate) const C_7: Scalar = 0.155;
pub(crate) const C_8: Scalar = 0.943;
pub(crate) const C_9: Scalar = 0.901802041735857;
pub(crate) const C_10: Scalar = 0.909;
pub(crate) const C_11: Scalar = 0.94;

pub(crate) const A_2_1: Scalar = 0.05;
pub(crate) const A_3_1: Scalar = -0.0069931640625;
pub(crate) const A_3_2: Scalar = 0.1135556640625;
pub(crate) const A_4_1: Scalar = 0.0399609375;
pub(crate) const A_4_3: Scalar = 0.1198828125;
pub(crate) const A_5_1: Scalar = 0.36139756280045754;
pub(crate) const A_5_3: Scalar = -1.3415240667004928;
pub(crate) const A_5_4: Scalar = 1.3701265039000352;
pub(crate) const A_6_1: Scalar = 0.049047202797202795;
pub(crate) const A_6_4: Scalar = 0.23509720422144048;
pub(crate) const A_6_5: Scalar = 0.18085559298135673;
pub(crate) const A_7_1: Scalar = 0.06169289044289044;
pub(crate) const A_7_4: Scalar = 0.11236568314640277;
pub(crate) const A_7_5: Scalar = -0.03885046071451367;
pub(crate) const A_7_6: Scalar = 0.01979188712522046;
pub(crate) const A_8_1: Scalar = -1.767630240222327;
pub(crate) const A_8_4: Scalar = -62.5;
pub(crate) const A_8_5: Scalar = -6.061889377376669;
pub(crate) const A_8_6: Scalar = 5.6508231982227635;
pub(crate) const A_8_7: Scalar = 65.62169641937624;
pub(crate) const A_9_1: Scalar = -1.1809450665549708;
pub(crate) const A_9_4: Scalar = -41.50473441114321;
pub(crate) const A_9_5: Scalar = -4.434438319103725;
pub(crate) const A_9_6: Scalar = 4.260408188586133;
pub(crate) const A_9_7: Scalar = 43.75364022446172;
pub(crate) const A_9_8: Scalar = 0.00787142548991231;
pub(crate) const A_10_1: Scalar = -1.2814059994414884;
pub(crate) const A_10_4: Scalar = -45.047139960139866;
pub(crate) const A_10_5: Scalar = -4.731362069449576;
pub(crate) const A_10_6: Scalar = 4.514967016593808;
pub(crate) const A_10_7: Scalar = 47.44909557172985;
pub(crate) const A_10_8: Scalar = 0.01059228297111661;
pub(crate) const A_10_9: Scalar = -0.0057468422638446166;
pub(crate) const A_11_1: Scalar = -1.7244701342624853;
pub(crate) const A_11_4: Scalar = -60.92349008483054;
pub(crate) const A_11_5: Scalar = -5.951518376222392;
pub(crate) const A_11_6: Scalar = 5.556523730698456;
pub(crate) const A_11_7: Scalar = 63.98301198033305;
pub(crate) const A_11_8: Scalar = 0.014642028250414961;
pub(crate) const A_11_9: Scalar = 0.06460408772358203;
pub(crate) const A_11_10: Scalar = -0.0793032316900888;
pub(crate) const A_12_1: Scalar = -3.301622667747079;
pub(crate) const A_12_4: Scalar = -118.01127235975251;
pub(crate) const A_12_5: Scalar = -10.141422388456112;
pub(crate) const A_12_6: Scalar = 9.139311332232058;
pub(crate) const A_12_7: Scalar = 123.37594282840426;
pub(crate) const A_12_8: Scalar = 4.62324437887458;
pub(crate) const A_12_9: Scalar = -3.3832777380682018;
pub(crate) const A_12_10: Scalar = 4.527592100324618;
pub(crate) const A_12_11: Scalar = -5.828495485811623;
pub(crate) const A_13_1: Scalar = -3.039515033766309;
pub(crate) const A_13_4: Scalar = -109.26086808941763;
pub(crate) const A_13_5: Scalar = -9.290642497400293;
pub(crate) const A_13_6: Scalar = 8.43050498176491;
pub(crate) const A_13_7: Scalar = 114.20100103783314;
pub(crate) const A_13_8: Scalar = -0.9637271342145479;
pub(crate) const A_13_9: Scalar = -5.0348840888021895;
pub(crate) const A_13_10: Scalar = 5.958130824002923;

pub(crate) const B_1: Scalar = 0.04427989419007951;
pub(crate) const B_6: Scalar = 0.3541049391724449;
pub(crate) const B_7: Scalar = 0.24796921549564377;
pub(crate) const B_8: Scalar = -15.694202038838085;
pub(crate) const B_9: Scalar = 25.084064965558564;
pub(crate) const B_10: Scalar = -31.738367786260277;
pub(crate) const B_11: Scalar = 22.938283273988784;
pub(crate) const B_12: Scalar = -0.2361324633071542;

pub(crate) const D_1: Scalar = -0.00003272103901028138;
pub(crate) const D_6: Scalar = -0.0005046250618777704;
pub(crate) const D_7: Scalar = 0.0001211723589784759;
pub(crate) const D_8: Scalar = -20.142336771313868;
pub(crate) const D_9: Scalar = 5.2371785994398286;
pub(crate) const D_10: Scalar = -8.156744408794658;
pub(crate) const D_11: Scalar = 22.938283273988784;
pub(crate) const D_12: Scalar = -0.2361324633071542;
pub(crate) const D_13: Scalar = 0.36016794372897754;

/// The Verner 8(7) tableau.
#[derive(Debug)]
pub struct Tableau;

impl ButcherTableau for Tableau {
    const STAGES: usize = 13;
    const ORDER: Scalar = 8.0;
    #[rustfmt::skip]
    const A: &'static [&'static [Scalar]] = &[
        &[],
        &[A_2_1],
        &[A_3_1, A_3_2],
        &[A_4_1, 0.0, A_4_3],
        &[A_5_1, 0.0, A_5_3, A_5_4],
        &[A_6_1, 0.0, 0.0, A_6_4, A_6_5],
        &[A_7_1, 0.0, 0.0, A_7_4, A_7_5, A_7_6],
        &[A_8_1, 0.0, 0.0, A_8_4, A_8_5, A_8_6, A_8_7],
        &[A_9_1, 0.0, 0.0, A_9_4, A_9_5, A_9_6, A_9_7, A_9_8],
        &[A_10_1, 0.0, 0.0, A_10_4, A_10_5, A_10_6, A_10_7, A_10_8, A_10_9],
        &[A_11_1, 0.0, 0.0, A_11_4, A_11_5, A_11_6, A_11_7, A_11_8, A_11_9, A_11_10],
        &[A_12_1, 0.0, 0.0, A_12_4, A_12_5, A_12_6, A_12_7, A_12_8, A_12_9, A_12_10, A_12_11],
        &[A_13_1, 0.0, 0.0, A_13_4, A_13_5, A_13_6, A_13_7, A_13_8, A_13_9, A_13_10, 0.0, 0.0],
    ];
    const C: &'static [Scalar] = &[
        0.0, C_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9, C_10, C_11, 1.0, 1.0,
    ];
    #[rustfmt::skip]
    const B: &'static [Scalar] =
        &[B_1, 0.0, 0.0, 0.0, 0.0, B_6, B_7, B_8, B_9, B_10, B_11, B_12, 0.0];
}

impl EmbeddedTableau for Tableau {
    #[rustfmt::skip]
    const D: &'static [Scalar] =
        &[D_1, 0.0, 0.0, 0.0, 0.0, D_6, D_7, D_8, D_9, D_10, D_11, D_12, D_13];
}

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
    /// Norm type for error evaluation.
    pub error_norm: Norm,
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
            error_norm: Norm::Chebyshev,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for Verner8
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> VariableStep<T> for Verner8 {
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

impl<Y, U, V, T> Explicit<Y, U, V, T> for Verner8
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 13;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> VariableStepExplicit<Y, U, V, T> for Verner8
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
    fn error(&self, dt: Quantity<T>, k: &[Derivative<Y, T>]) -> Result<Scalar, String> {
        self.error_from_tableau::<Tableau>(dt, k)
    }
    fn slopes(
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        Self::slopes_from_tableau::<Tableau>(function, y, t, dt, k, y_trial)
    }
}

impl<Y, U, V, T> InterpolateSolution<Y, U, V, T> for Verner8
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
        _dydtp: &V,
        _k_sol: &[V],
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
    ) -> Result<(U, V), IntegrationError> {
        Self::interpolate_variable_step(time, tp, yp, function)
    }
}
