//! Autodiff-backed elastic constitutive models (`std::autodiff` / Enzyme).
//!
//! A model that supplies `#[autodiff]`-differentiable scalar stress kernels
//! ([`AutodiffElastic`]) gets the full `Elastic` API by wrapping it in
//! [`Autodiff`]. `Hyperelastic` builds on this in
//! [`hyperelastic::autodiff`](crate::constitutive::solid::hyperelastic::autodiff).
//! Maintained models keep their hand-written stress and tangent.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub mod saint_venant_kirchhoff;

pub use crate::constitutive::autodiff::Autodiff;
pub use saint_venant_kirchhoff::AutodiffSaintVenantKirchhoff;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic},
    },
    math::{Quantity, TensorRank2, TensorRank4},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::Stress,
};
use std::fmt::Debug;

/// An elastic model given as `#[autodiff]`-differentiable scalar kernels over a
/// row-major deformation gradient: three stress measures, and by forward mode
/// over each (one `F_kL` direction per call) their `d(stress)/dF` tangents.
pub trait AutodiffElastic {
    fn parameters(&self) -> [f64; 2];
    fn bulk_modulus(&self) -> Quantity<Stress>;
    fn shear_modulus(&self) -> Quantity<Stress>;
    fn cauchy(parameters: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]);
    fn piola(parameters: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]);
    fn second_piola(parameters: &[f64; 2], f: &[f64; 9], out: &mut [f64; 9]);
    fn cauchy_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn piola_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn second_piola_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
}

impl<M> Solid for Autodiff<M>
where
    M: AutodiffElastic + Clone + Debug,
{
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.0.bulk_modulus()
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.0.shear_modulus()
    }
}

impl<M> Elastic for Autodiff<M>
where
    M: AutodiffElastic + Clone + Debug,
{
    fn cauchy_stress(&self, f: &DeformationGradient) -> Result<CauchyStress, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(stress(&self.0.parameters(), f, M::cauchy))
    }
    fn first_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(stress(&self.0.parameters(), f, M::piola))
    }
    fn second_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(stress(&self.0.parameters(), f, M::second_piola))
    }
    fn cauchy_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(&self.0.parameters(), f, M::cauchy_tangent))
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(&self.0.parameters(), f, M::piola_tangent))
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(&self.0.parameters(), f, M::second_piola_tangent))
    }
}

fn stress<I, J>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64; 2], &[f64; 9], &mut [f64; 9]),
) -> TensorRank2<3, I, J, Stress> {
    let f = deformation_gradient.flatten();
    let mut out = [0.0; 9];
    kernel(parameters, &f, &mut out);
    TensorRank2::unflatten(out)
}

fn tangent<I, J, K, L>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64; 2], &[f64; 9], &[f64; 9], &mut [f64; 9], &mut [f64; 9]),
) -> TensorRank4<3, I, J, K, L, Stress> {
    let f = deformation_gradient.flatten();
    let mut c = [0.0; 81];
    for k in 0..3 {
        for l in 0..3 {
            let mut df = [0.0; 9];
            df[3 * k + l] = 1.0;
            let (mut primal, mut seed) = ([0.0; 9], [0.0; 9]);
            kernel(parameters, &f, &df, &mut primal, &mut seed);
            for i in 0..3 {
                for j in 0..3 {
                    c[27 * i + 9 * j + 3 * k + l] = seed[3 * i + j];
                }
            }
        }
    }
    TensorRank4::unflatten(c)
}
