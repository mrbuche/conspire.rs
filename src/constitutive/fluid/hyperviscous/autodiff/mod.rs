//! Autodiff-backed hyperviscous constitutive models (`std::autodiff` / Enzyme).
//!
//! A model that supplies its viscous dissipation potential $`\psi(\mathbf{F},\dot{\mathbf{F}})`$
//! as an `#[autodiff]`-differentiable scalar kernel ([`AutodiffHyperviscous`]) gets
//! the full `Viscous` + `Hyperviscous` API by wrapping it in [`Autodiff`]: the
//! first Piola-Kirchhoff viscous stress is reverse mode over $`\psi`$ w.r.t.
//! $`\dot{\mathbf{F}}`$, and the rate tangent stiffnesses are forward mode over
//! the stress kernels. Composing `Canonical<E, Autodiff<V>>` then yields a
//! fully autodiff viscoelastic solid. Maintained models keep their hand-written
//! stress and tangent.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub mod newtonian;

pub use crate::constitutive::autodiff::Autodiff;
pub use newtonian::AutodiffNewtonian;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{hyperviscous::Hyperviscous, viscous::Viscous},
    },
    math::{Quantity, TensorRank2, TensorRank4},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        SecondPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffStress,
    },
    units::{Dissipation, Stress, Viscosity},
};
use std::fmt::Debug;

/// A hyperviscous model given as an `#[autodiff]`-differentiable scalar viscous
/// dissipation kernel over a row-major deformation gradient and its rate: three
/// viscous stress measures obtained by reverse mode w.r.t. `f_dot`, and by
/// forward mode over each (one `Fdot_kL` direction per call) their
/// `d(stress)/dFdot` rate tangents.
pub trait AutodiffHyperviscous {
    fn parameters(&self) -> [f64; 2];
    fn bulk_viscosity(&self) -> Quantity<Viscosity>;
    fn shear_viscosity(&self) -> Quantity<Viscosity>;
    fn dissipation(parameters: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9]) -> f64;
    fn viscous_cauchy(parameters: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9], out: &mut [f64; 9]);
    fn viscous_piola(parameters: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9], out: &mut [f64; 9]);
    fn viscous_second_piola(
        parameters: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        out: &mut [f64; 9],
    );
    fn viscous_cauchy_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn viscous_piola_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
    fn viscous_second_piola_tangent(
        parameters: &[f64; 2],
        f: &[f64; 9],
        f_dot: &[f64; 9],
        df_dot: &[f64; 9],
        primal: &mut [f64; 9],
        seed: &mut [f64; 9],
    );
}

impl<M> Viscous for Autodiff<M>
where
    M: AutodiffHyperviscous + Clone + Debug,
{
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.0.bulk_viscosity()
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.0.shear_viscosity()
    }
    fn viscous_cauchy_stress(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(stress(&self.0.parameters(), f, f_dot, M::viscous_cauchy))
    }
    fn viscous_first_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(stress(&self.0.parameters(), f, f_dot, M::viscous_piola))
    }
    fn viscous_second_piola_kirchhoff_stress(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(stress(
            &self.0.parameters(),
            f,
            f_dot,
            M::viscous_second_piola,
        ))
    }
    fn viscous_cauchy_rate_tangent_stiffness(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        Ok(tangent(
            &self.0.parameters(),
            f,
            f_dot,
            M::viscous_cauchy_tangent,
        ))
    }
    fn viscous_first_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        Ok(tangent(
            &self.0.parameters(),
            f,
            f_dot,
            M::viscous_piola_tangent,
        ))
    }
    fn viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        Ok(tangent(
            &self.0.parameters(),
            f,
            f_dot,
            M::viscous_second_piola_tangent,
        ))
    }
}

impl<M> Hyperviscous for Autodiff<M>
where
    M: AutodiffHyperviscous + Clone + Debug,
{
    fn viscous_dissipation(
        &self,
        f: &DeformationGradient,
        f_dot: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(Quantity::new(M::dissipation(
            &self.0.parameters(),
            &f.flatten(),
            &f_dot.flatten(),
        )))
    }
}

fn stress<I, J>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    deformation_gradient_rate: &DeformationGradientRate,
    kernel: fn(&[f64; 2], &[f64; 9], &[f64; 9], &mut [f64; 9]),
) -> TensorRank2<3, I, J, Stress> {
    let (f, f_dot) = (
        deformation_gradient.flatten(),
        deformation_gradient_rate.flatten(),
    );
    let mut out = [0.0; 9];
    kernel(parameters, &f, &f_dot, &mut out);
    TensorRank2::unflatten(out)
}

fn tangent<I, J, K, L>(
    parameters: &[f64; 2],
    deformation_gradient: &DeformationGradient,
    deformation_gradient_rate: &DeformationGradientRate,
    kernel: fn(&[f64; 2], &[f64; 9], &[f64; 9], &[f64; 9], &mut [f64; 9], &mut [f64; 9]),
) -> TensorRank4<3, I, J, K, L, Viscosity> {
    let (f, f_dot) = (
        deformation_gradient.flatten(),
        deformation_gradient_rate.flatten(),
    );
    let mut c = [0.0; 81];
    for k in 0..3 {
        for l in 0..3 {
            let mut df_dot = [0.0; 9];
            df_dot[3 * k + l] = 1.0;
            let (mut primal, mut seed) = ([0.0; 9], [0.0; 9]);
            kernel(parameters, &f, &f_dot, &df_dot, &mut primal, &mut seed);
            for i in 0..3 {
                for j in 0..3 {
                    c[27 * i + 9 * j + 3 * k + l] = seed[3 * i + j];
                }
            }
        }
    }
    TensorRank4::unflatten(c)
}
