//! Hyperviscous fluid constitutive models.

mod newtonian;
mod saint_venant_kirchhoff;

pub use self::{newtonian::Newtonian, saint_venant_kirchhoff::SaintVenantKirchhoff};

use crate::{
    constitutive::{ConstitutiveError, fluid::viscous::Viscous},
    math::{Quantity, Scalar},
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::Dissipation,
};

const TWO_THIRDS: Scalar = 2.0 / 3.0;

/// Required methods for hyperviscous fluid constitutive models.
pub trait Hyperviscous
where
    Self: Viscous,
{
    /// Calculates and returns the viscous dissipation.
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError>;
}
