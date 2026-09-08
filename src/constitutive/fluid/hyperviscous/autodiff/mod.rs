//! Autodiff-backed hyperviscous constitutive models (`std::autodiff` / Enzyme).
//!
//! Builds on [`viscous::autodiff`](crate::constitutive::fluid::viscous::autodiff):
//! a model supplies its viscous dissipation potential
//! $`\psi(\mathbf{F},\dot{\mathbf{F}})`$ as a scalar kernel plus the
//! [`AutodiffViscous`] stress kernels (the first-Piola pair being reverse mode
//! over $`\psi`$ w.r.t. $`\dot{\mathbf{F}}`$), and wrapping it in [`Autodiff`]
//! provides the full `Hyperviscous` API. Composing `Canonical<E, Autodiff<V>>`
//! then yields a fully autodiff viscoelastic solid. Maintained models keep their
//! hand-written stress and tangent.

pub mod newtonian;

pub use crate::constitutive::{autodiff::Autodiff, fluid::viscous::autodiff::AutodiffViscous};
pub use newtonian::AutodiffNewtonian;

use crate::{
    constitutive::{ConstitutiveError, fluid::hyperviscous::Hyperviscous},
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::Dissipation,
};
use std::fmt::Debug;

/// An [`AutodiffViscous`] model that also exposes its viscous dissipation
/// potential as a scalar kernel (its first-Piola viscous stress kernel being
/// reverse mode over this w.r.t. `f_dot`).
pub trait AutodiffHyperviscous: AutodiffViscous {
    fn dissipation(parameters: &[f64; 2], f: &[f64; 9], f_dot: &[f64; 9]) -> f64;
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
