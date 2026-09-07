//! Autodiff-backed elastic constitutive models (`std::autodiff` / Enzyme).
//!
//! A model that supplies `#[autodiff]`-differentiable scalar stress kernels
//! ([`AutodiffElastic`]) gets the full `Elastic` API by wrapping it in
//! [`Autodiff`]. `Hyperelastic` builds on this in
//! [`hyperelastic::autodiff`](crate::constitutive::solid::hyperelastic::autodiff).
//! Maintained models keep their hand-written stress and tangent.

#![allow(clippy::needless_range_loop)]

#[cfg(test)]
pub(crate) mod test;

pub mod saint_venant_kirchhoff;

pub use crate::constitutive::autodiff::Autodiff;
pub use saint_venant_kirchhoff::AutodiffSaintVenantKirchhoff;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic},
    },
    math::{Quantity, TensorArray},
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

pub(crate) fn flatten(deformation_gradient: &DeformationGradient) -> [f64; 9] {
    let a = deformation_gradient.as_array();
    [
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ]
}

pub(crate) fn determinant(f: &[f64; 9]) -> f64 {
    f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6])
}

pub(crate) fn push_cauchy(p: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let jacobian = determinant(f);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] =
                (p[3 * i] * f[3 * j] + p[3 * i + 1] * f[3 * j + 1] + p[3 * i + 2] * f[3 * j + 2])
                    / jacobian;
        }
    }
}

pub(crate) fn inverse(f: &[f64; 9]) -> [f64; 9] {
    let jacobian = determinant(f);
    [
        (f[4] * f[8] - f[5] * f[7]) / jacobian,
        (f[2] * f[7] - f[1] * f[8]) / jacobian,
        (f[1] * f[5] - f[2] * f[4]) / jacobian,
        (f[5] * f[6] - f[3] * f[8]) / jacobian,
        (f[0] * f[8] - f[2] * f[6]) / jacobian,
        (f[2] * f[3] - f[0] * f[5]) / jacobian,
        (f[3] * f[7] - f[4] * f[6]) / jacobian,
        (f[1] * f[6] - f[0] * f[7]) / jacobian,
        (f[0] * f[4] - f[1] * f[3]) / jacobian,
    ]
}

pub(crate) fn push_second_piola(p: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let f_inverse = inverse(f);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = f_inverse[3 * i] * p[j]
                + f_inverse[3 * i + 1] * p[3 + j]
                + f_inverse[3 * i + 2] * p[6 + j];
        }
    }
}

pub(crate) fn push_first_piola(sigma: &[f64; 9], f: &[f64; 9], out: &mut [f64; 9]) {
    let (jacobian, f_inverse) = (determinant(f), inverse(f));
    for i in 0..3 {
        for k in 0..3 {
            out[3 * i + k] = jacobian
                * (sigma[3 * i] * f_inverse[3 * k]
                    + sigma[3 * i + 1] * f_inverse[3 * k + 1]
                    + sigma[3 * i + 2] * f_inverse[3 * k + 2]);
        }
    }
}

fn unflatten_stress<T: From<[[f64; 3]; 3]>>(m: [f64; 9]) -> T {
    T::from([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]])
}

fn unflatten_tangent<T: From<[[[[f64; 3]; 3]; 3]; 3]>>(c: [[[[f64; 3]; 3]; 3]; 3]) -> T {
    T::from(c)
}

fn stress<T: From<[[f64; 3]; 3]>>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64; 2], &[f64; 9], &mut [f64; 9]),
) -> T {
    let f = flatten(deformation_gradient);
    let mut out = [0.0; 9];
    kernel(parameters, &f, &mut out);
    unflatten_stress(out)
}

fn tangent<T: From<[[[[f64; 3]; 3]; 3]; 3]>>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    kernel: fn(&[f64; 2], &[f64; 9], &[f64; 9], &mut [f64; 9], &mut [f64; 9]),
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
