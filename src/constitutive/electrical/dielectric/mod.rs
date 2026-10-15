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
    constitutive::{ConstitutiveError, electrical::Electrical},
    math::Quantity,
    mechanics::{ElectricDisplacement, ElectricField, ElectricTangent},
    units::EnergyDensity,
};

pub use linear::Linear;

/// Required methods for dielectric constitutive models.
pub trait Dielectric
where
    Self: Electrical,
{
    /// Calculates and returns the potential.
    fn potential(
        &self,
        electric_field: &ElectricField,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
    /// Calculates and returns the electric displacement.
    fn electric_displacement(
        &self,
        electric_field: &ElectricField,
    ) -> Result<ElectricDisplacement, ConstitutiveError>;
    /// Calculates and returns the tangent to the electric displacement.
    fn electric_tangent(
        &self,
        electric_field: &ElectricField,
    ) -> Result<ElectricTangent, ConstitutiveError>;
}
