//! Autodiff-backed elastic constitutive models (`std::autodiff` / Enzyme).
//!
//! Opt-in, `--features autodiff`; needs a nightly `rustc` with the Enzyme
//! backend (`rustup component add enzyme`) and a fat-LTO profile:
//!
//! ```text
//! RUSTFLAGS="-Zautodiff=Enable" cargo +nightly test --release --features autodiff -j1
//! ```
//!
//! A model that supplies `#[autodiff]`-differentiable scalar stress kernels
//! ([`AutodiffElastic`]) gets the full `Elastic` API by wrapping it in
//! [`Autodiff`]. `Hyperelastic` builds on this in
//! [`hyperelastic::autodiff`](crate::constitutive::solid::hyperelastic::autodiff).
//! Maintained models keep their hand-written stress and tangent.
//!
//! Deformation gradients flatten row-major (`f[3 * i + j] = F_iJ`).

#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic},
    },
    math::Quantity,
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::Stress,
};
use std::fmt::Debug;

pub(crate) fn flatten(deformation_gradient: &DeformationGradient) -> [f64; 9] {
    let mut f = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            f[3 * i + j] = deformation_gradient[i][j].value();
        }
    }
    f
}

fn unflatten_stress<T: From<[[f64; 3]; 3]>>(m: [f64; 9]) -> T {
    T::from([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]])
}

fn unflatten_tangent<T: From<[[[[f64; 3]; 3]; 3]; 3]>>(c: [[[[f64; 3]; 3]; 3]; 3]) -> T {
    T::from(c)
}

fn stress<T: From<[[f64; 3]; 3]>>(
    parameters: &[f64],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64], &[f64; 9], &mut [f64; 9]),
) -> T {
    let f = flatten(deformation_gradient);
    let mut out = [0.0; 9];
    kernel(parameters, &f, &mut out);
    unflatten_stress(out)
}

fn tangent<T: From<[[[[f64; 3]; 3]; 3]; 3]>>(
    parameters: &[f64],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64], &[f64; 9], &[f64; 9], &mut [f64; 9], &mut [f64; 9]),
) -> T {
    let f = flatten(deformation_gradient);
    let mut c = [[[[0.0; 3]; 3]; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            let mut df = [0.0; 9];
            df[3 * k + l] = 1.0;
            let (mut primal, mut seed) = ([0.0; 9], [0.0; 9]);
            kernel(parameters, &f, &df, &mut primal, &mut seed);
            for i in 0..3 {
                for j in 0..3 {
                    c[i][j][k][l] = seed[3 * i + j];
                }
            }
        }
    }
    unflatten_tangent(c)
}

/// An elastic model given as `#[autodiff]`-differentiable scalar kernels over a
/// row-major deformation gradient: three stress measures, and by forward mode
/// over each (one `F_kL` direction per call) their `d(stress)/dF` tangents.
pub trait AutodiffElastic {
    /// Scalar material parameters, passed first to every kernel.
    type Parameters: AsRef<[f64]>;
    fn parameters(&self) -> Self::Parameters;
    fn bulk_modulus(&self) -> Quantity<Stress>;
    fn shear_modulus(&self) -> Quantity<Stress>;
    fn cauchy(parameters: &[f64], f: &[f64; 9], out: &mut [f64; 9]);
    fn piola(parameters: &[f64], f: &[f64; 9], out: &mut [f64; 9]);
    fn second_piola(parameters: &[f64], f: &[f64; 9], out: &mut [f64; 9]);
    fn cauchy_tangent(
        parameters: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn piola_tangent(
        parameters: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn second_piola_tangent(
        parameters: &[f64],
        f: &[f64; 9],
        df: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
}

/// Gives any [`AutodiffElastic`] the full `Elastic` API (and, for an
/// [`AutodiffHyperelastic`](crate::constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic),
/// `Hyperelastic`), every stress and tangent obtained by autodiff of its kernels.
#[derive(Clone, Debug)]
pub struct Autodiff<M>(pub M);

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
        Ok(stress(self.0.parameters().as_ref(), f, M::cauchy))
    }
    fn first_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(stress(self.0.parameters().as_ref(), f, M::piola))
    }
    fn second_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(stress(self.0.parameters().as_ref(), f, M::second_piola))
    }
    fn cauchy_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(self.0.parameters().as_ref(), f, M::cauchy_tangent))
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(self.0.parameters().as_ref(), f, M::piola_tangent))
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        f: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(tangent(
            self.0.parameters().as_ref(),
            f,
            M::second_piola_tangent,
        ))
    }
}
