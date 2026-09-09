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

pub(crate) const C_2: Scalar = 0.03462;
pub(crate) const C_3: Scalar = 0.097_024_350_638_780_44;
pub(crate) const C_4: Scalar = 0.145_536_525_958_170_67;
pub(crate) const C_5: Scalar = 0.561;
pub(crate) const C_6: Scalar = 0.229_007_911_590_485;
pub(crate) const C_7: Scalar = 0.544_992_088_409_515;
pub(crate) const C_8: Scalar = 0.645;
pub(crate) const C_9: Scalar = 0.48375;
pub(crate) const C_10: Scalar = 0.06757;
pub(crate) const C_11: Scalar = 0.2500;
pub(crate) const C_12: Scalar = 0.659_065_061_873_099_9;
pub(crate) const C_13: Scalar = 0.8206;
pub(crate) const C_14: Scalar = 0.9012;

pub(crate) const A_2_1: Scalar = 0.03462;
pub(crate) const A_3_1: Scalar = -0.03893354388572875;
pub(crate) const A_3_2: Scalar = 0.13595789452450918;
pub(crate) const A_4_1: Scalar = 0.03638413148954267;
pub(crate) const A_4_3: Scalar = 0.10915239446862801;
pub(crate) const A_5_1: Scalar = 2.0257639143939694;
pub(crate) const A_5_3: Scalar = -7.638023836496291;
pub(crate) const A_5_4: Scalar = 6.173259922102322;
pub(crate) const A_6_1: Scalar = 0.05112275589406061;
pub(crate) const A_6_4: Scalar = 0.17708237945550218;
pub(crate) const A_6_5: Scalar = 0.0008027762409222536;
pub(crate) const A_7_1: Scalar = 0.13160063579752163;
pub(crate) const A_7_4: Scalar = -0.2957276252669636;
pub(crate) const A_7_5: Scalar = 0.08781378035642955;
pub(crate) const A_7_6: Scalar = 0.6213052975225274;
pub(crate) const A_8_1: Scalar = 0.07166666666666667;
pub(crate) const A_8_6: Scalar = 0.33055335789153195;
pub(crate) const A_8_7: Scalar = 0.2427799754418014;
pub(crate) const A_9_1: Scalar = 0.071806640625;
pub(crate) const A_9_6: Scalar = 0.3294380283228177;
pub(crate) const A_9_7: Scalar = 0.1165190029271823;
pub(crate) const A_9_8: Scalar = -0.034013671875;
pub(crate) const A_10_1: Scalar = 0.04836757646340646;
pub(crate) const A_10_6: Scalar = 0.03928989925676164;
pub(crate) const A_10_7: Scalar = 0.10547409458903446;
pub(crate) const A_10_8: Scalar = -0.021438652846483126;
pub(crate) const A_10_9: Scalar = -0.10412291746271944;
pub(crate) const A_11_1: Scalar = -0.026645614872014785;
pub(crate) const A_11_6: Scalar = 0.03333333333333333;
pub(crate) const A_11_7: Scalar = -0.1631072244872467;
pub(crate) const A_11_8: Scalar = 0.03396081684127761;
pub(crate) const A_11_9: Scalar = 0.1572319413814626;
pub(crate) const A_11_10: Scalar = 0.21522674780318796;
pub(crate) const A_12_1: Scalar = 0.03689009248708622;
pub(crate) const A_12_6: Scalar = -0.1465181576725543;
pub(crate) const A_12_7: Scalar = 0.2242577768172024;
pub(crate) const A_12_8: Scalar = 0.02294405717066073;
pub(crate) const A_12_9: Scalar = -0.0035850052905728597;
pub(crate) const A_12_10: Scalar = 0.08669223316444385;
pub(crate) const A_12_11: Scalar = 0.43838406519683376;
pub(crate) const A_13_1: Scalar = -0.4866012215113341;
pub(crate) const A_13_6: Scalar = -6.304602650282853;
pub(crate) const A_13_7: Scalar = -0.2812456182894729;
pub(crate) const A_13_8: Scalar = -2.679019236219849;
pub(crate) const A_13_9: Scalar = 0.5188156639241577;
pub(crate) const A_13_10: Scalar = 1.3653531876033418;
pub(crate) const A_13_11: Scalar = 5.8850910885039465;
pub(crate) const A_13_12: Scalar = 2.8028087862720628;
pub(crate) const A_14_1: Scalar = 0.4185367457753472;
pub(crate) const A_14_6: Scalar = 6.724547581906459;
pub(crate) const A_14_7: Scalar = -0.42544428016461133;
pub(crate) const A_14_8: Scalar = 3.3432791530012653;
pub(crate) const A_14_9: Scalar = 0.6170816631175374;
pub(crate) const A_14_10: Scalar = -0.9299661239399329;
pub(crate) const A_14_11: Scalar = -6.099948804751011;
pub(crate) const A_14_12: Scalar = -3.002206187889399;
pub(crate) const A_14_13: Scalar = 0.2553202529443446;
pub(crate) const A_15_1: Scalar = -0.7793740861228848;
pub(crate) const A_15_6: Scalar = -13.937342538107776;
pub(crate) const A_15_7: Scalar = 1.2520488533793563;
pub(crate) const A_15_8: Scalar = -14.691500408016868;
pub(crate) const A_15_9: Scalar = -0.494705058533141;
pub(crate) const A_15_10: Scalar = 2.2429749091462368;
pub(crate) const A_15_11: Scalar = 13.367893803828643;
pub(crate) const A_15_12: Scalar = 14.396650486650687;
pub(crate) const A_15_13: Scalar = -0.79758133317768;
pub(crate) const A_15_14: Scalar = 0.4409353709534278;
pub(crate) const A_16_1: Scalar = 2.0580513374668867;
pub(crate) const A_16_6: Scalar = 22.357937727968032;
pub(crate) const A_16_7: Scalar = 0.9094981099755646;
pub(crate) const A_16_8: Scalar = 35.89110098240264;
pub(crate) const A_16_9: Scalar = -3.442515027624454;
pub(crate) const A_16_10: Scalar = -4.865481358036369;
pub(crate) const A_16_11: Scalar = -18.909803813543427;
pub(crate) const A_16_12: Scalar = -34.26354448030452;
pub(crate) const A_16_13: Scalar = 1.2647565216956427;

pub(crate) const B_1: Scalar = 0.014611976858423152;
pub(crate) const B_8: Scalar = -0.3915211862331339;
pub(crate) const B_9: Scalar = 0.23109325002895065;
pub(crate) const B_10: Scalar = 0.12747667699928525;
pub(crate) const B_11: Scalar = 0.2246434176204158;
pub(crate) const B_12: Scalar = 0.5684352689748513;
pub(crate) const B_13: Scalar = 0.058258715572158275;
pub(crate) const B_14: Scalar = 0.13643174034822156;
pub(crate) const B_15: Scalar = 0.030570139830827976;

pub(crate) const D_1: Scalar = -0.005357988290444578;
pub(crate) const D_8: Scalar = -2.583020491182464;
pub(crate) const D_9: Scalar = 0.14252253154686625;
pub(crate) const D_10: Scalar = 0.013420653512688676;
pub(crate) const D_11: Scalar = -0.02867296291409493;
pub(crate) const D_12: Scalar = 2.624999655215792;
pub(crate) const D_13: Scalar = -0.2825509643291537;
pub(crate) const D_14: Scalar = 0.13643174034822156;
pub(crate) const D_15: Scalar = 0.030570139830827976;
pub(crate) const D_16: Scalar = -0.04834231373823958;

/// The Verner 9(8) tableau.
#[derive(Debug)]
pub struct Tableau;

impl ButcherTableau for Tableau {
    const STAGES: usize = 16;
    const ORDER: Scalar = 9.0;
    #[rustfmt::skip]
    const A: &'static [&'static [Scalar]] = &[
        &[],
        &[A_2_1],
        &[A_3_1, A_3_2],
        &[A_4_1, 0.0, A_4_3],
        &[A_5_1, 0.0, A_5_3, A_5_4],
        &[A_6_1, 0.0, 0.0, A_6_4, A_6_5],
        &[A_7_1, 0.0, 0.0, A_7_4, A_7_5, A_7_6],
        &[A_8_1, 0.0, 0.0, 0.0, 0.0, A_8_6, A_8_7],
        &[A_9_1, 0.0, 0.0, 0.0, 0.0, A_9_6, A_9_7, A_9_8],
        &[A_10_1, 0.0, 0.0, 0.0, 0.0, A_10_6, A_10_7, A_10_8, A_10_9],
        &[A_11_1, 0.0, 0.0, 0.0, 0.0, A_11_6, A_11_7, A_11_8, A_11_9, A_11_10],
        &[A_12_1, 0.0, 0.0, 0.0, 0.0, A_12_6, A_12_7, A_12_8, A_12_9, A_12_10, A_12_11],
        &[A_13_1, 0.0, 0.0, 0.0, 0.0, A_13_6, A_13_7, A_13_8, A_13_9, A_13_10, A_13_11, A_13_12],
        &[A_14_1, 0.0, 0.0, 0.0, 0.0, A_14_6, A_14_7, A_14_8, A_14_9, A_14_10, A_14_11, A_14_12, A_14_13],
        &[A_15_1, 0.0, 0.0, 0.0, 0.0, A_15_6, A_15_7, A_15_8, A_15_9, A_15_10, A_15_11, A_15_12, A_15_13, A_15_14],
        &[A_16_1, 0.0, 0.0, 0.0, 0.0, A_16_6, A_16_7, A_16_8, A_16_9, A_16_10, A_16_11, A_16_12, A_16_13, 0.0, 0.0],
    ];
    #[rustfmt::skip]
    const C: &'static [Scalar] =
        &[0.0, C_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9, C_10, C_11, C_12, C_13, C_14, 1.0, 1.0];
    #[rustfmt::skip]
    const B: &'static [Scalar] =
        &[B_1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15, 0.0];
}

impl EmbeddedTableau for Tableau {
    #[rustfmt::skip]
    const D: &'static [Scalar] =
        &[D_1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, D_8, D_9, D_10, D_11, D_12, D_13, D_14, D_15, D_16];
}

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
    /// Norm type for error evaluation.
    pub error_norm: Norm,
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
            error_norm: Norm::Chebyshev,
        }
    }
}

impl<Y, U> OdeIntegrator<Y, U> for Verner9
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> VariableStep<T> for Verner9 {
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

impl<Y, U, V, T> Explicit<Y, U, V, T> for Verner9
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 16;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> VariableStepExplicit<Y, U, V, T> for Verner9
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
}

impl<Y, U, V, T> InterpolateSolution<Y, U, V, T> for Verner9
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
