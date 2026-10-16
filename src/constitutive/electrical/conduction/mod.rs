//! Electrical conduction constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod ohmic;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    mechanics::{CurrentDensity, CurrentTangent, ElectricField},
    units::Current,
};

pub use ohmic::Ohmic;

/// Required methods for electrical conduction constitutive models.
pub trait ElectricalConduction
where
    Self: Electrical + Diffusion<Balance = Current>,
{
    /// Calculates and returns the current density.
    fn current_density(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentDensity, ConstitutiveError> {
        self.flux(potential_gradient)
    }
    /// Calculates and returns the tangent to the current density.
    fn current_tangent(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentTangent, ConstitutiveError> {
        self.tangent(potential_gradient)
    }
}

impl<C> ElectricalConduction for C where C: Electrical + Diffusion<Balance = Current> {}
