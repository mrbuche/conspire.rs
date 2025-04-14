//! Hyperelastic constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(test)]
pub mod test;

mod arruda_boyce;
// mod fung;
// mod gent;
// mod mooney_rivlin;
// mod neo_hookean;
// mod saint_venant_kirchhoff;
// mod yeoh;

pub use self::{
    arruda_boyce::ArrudaBoyce, //fung::Fung, gent::Gent, mooney_rivlin::MooneyRivlin,
                               //neo_hookean::NeoHookean, saint_venant_kirchhoff::SaintVenantKirchhoff, yeoh::Yeoh,
};
use super::{elastic::Elastic, *};
use std::fmt::Debug;

/// Required methods for hyperelastic constitutive models.
pub trait Hyperelastic<P>
where
    Self: Elastic<P>,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError>;
}
