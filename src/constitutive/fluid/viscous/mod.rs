//! Viscous fluid constitutive models.

mod newtonian;
mod saint_venant_kirchhoff;

pub use self::{newtonian::Newtonian, saint_venant_kirchhoff::SaintVenantKirchhoff};

use crate::{
    constitutive::ConstitutiveError,
    math::{ContractFirstSecondWithSecond, ContractSecondWithFirst, Quantity, Rank2, Scalar},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        SecondPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffStress,
    },
    units::Viscosity,
};

const TWO_THIRDS: Scalar = 2.0 / 3.0;

/// Required methods for viscous fluid constitutive models.
///
/// The methods return the viscous contribution only; a concrete model implements
/// one stress measure and its rate tangent, and the rest follow.
pub trait Viscous {
    /// Returns the bulk viscosity.
    fn bulk_viscosity(&self) -> Quantity<Viscosity>;
    /// Returns the shear viscosity.
    fn shear_viscosity(&self) -> Quantity<Viscosity>;
    /// Calculates and returns the viscous Cauchy stress.
    fn viscous_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.viscous_second_piola_kirchhoff_stress(
                deformation_gradient,
                deformation_gradient_rate,
            )?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the rate tangent stiffness associated with the viscous Cauchy stress.
    fn viscous_cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        Ok(self
            .viscous_second_piola_kirchhoff_rate_tangent_stiffness(
                deformation_gradient,
                deformation_gradient_rate,
            )?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the viscous first Piola-Kirchhoff stress.
    fn viscous_first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.viscous_cauchy_stress(deformation_gradient, deformation_gradient_rate)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the rate tangent stiffness associated with the viscous first Piola-Kirchhoff stress.
    fn viscous_first_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        Ok(self
            .viscous_cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_second_with_first(&deformation_gradient.inverse_transpose())
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the viscous second Piola-Kirchhoff stress.
    fn viscous_second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.viscous_cauchy_stress(deformation_gradient, deformation_gradient_rate)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the rate tangent stiffness associated with the viscous second Piola-Kirchhoff stress.
    fn viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse = deformation_gradient.inverse();
        Ok(self
            .viscous_cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_first_second_with_second(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant())
    }
}
