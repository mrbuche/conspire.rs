//! Dielectric constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod linear;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    mechanics::{ElectricDisplacement, ElectricField, ElectricTangent},
    units::ElectricPotential,
};

pub use linear::Linear;

/// Required methods for dielectric constitutive models.
pub trait Dielectric
where
    Self: Electrical + Diffusion<Field = ElectricPotential>,
{
    /// Calculates and returns the electric displacement.
    fn electric_displacement(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<ElectricDisplacement, ConstitutiveError> {
        self.flux(potential_gradient)
    }
    /// Calculates and returns the tangent to the electric displacement.
    fn electric_tangent(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<ElectricTangent, ConstitutiveError> {
        self.tangent(potential_gradient)
    }
}

impl<C> Dielectric for C where C: Electrical + Diffusion<Field = ElectricPotential> {}
