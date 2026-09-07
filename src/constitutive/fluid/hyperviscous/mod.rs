//! Hyperviscous fluid constitutive models.

use crate::{
    constitutive::{ConstitutiveError, fluid::viscous::Viscous},
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::Dissipation,
};

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
