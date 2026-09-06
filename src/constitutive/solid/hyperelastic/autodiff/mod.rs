//! Autodiff-backed hyperelastic constitutive models (`std::autodiff` / Enzyme).
//!
//! Builds on [`elastic::autodiff`](crate::constitutive::solid::elastic::autodiff):
//! a model supplies its Helmholtz free energy density as a scalar kernel plus
//! the [`AutodiffElastic`] stress kernels (the first-Piola pair being reverse
//! mode over the energy), and wrapping it in [`Autodiff`] provides the full
//! `Hyperelastic` API. Maintained models keep their hand-written impls.

pub mod neo_hookean;
pub mod saint_venant_kirchhoff;

pub use self::{
    neo_hookean::AutodiffNeoHookean, saint_venant_kirchhoff::AutodiffSaintVenantKirchhoff,
};
pub use crate::constitutive::solid::elastic::autodiff::{Autodiff, AutodiffElastic};

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::autodiff::flatten, hyperelastic::Hyperelastic},
    },
    math::Quantity,
    mechanics::DeformationGradient,
    units::EnergyDensity,
};
use std::fmt::Debug;

/// An [`AutodiffElastic`] model that also exposes its Helmholtz free energy
/// density as a scalar kernel (its first-Piola stress kernel being reverse mode
/// over this).
pub trait AutodiffHyperelastic: AutodiffElastic {
    fn energy(parameters: &[f64; 2], f: &[f64; 9]) -> f64;
}

impl<M> Hyperelastic for Autodiff<M>
where
    M: AutodiffHyperelastic + Clone + Debug,
{
    fn helmholtz_free_energy_density(
        &self,
        f: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        self.jacobian(f)?;
        Ok(Quantity::new(M::energy(&self.0.parameters(), &flatten(f))))
    }
}
