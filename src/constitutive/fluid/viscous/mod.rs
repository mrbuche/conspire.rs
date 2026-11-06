//! Viscous fluid constitutive models.

#[cfg(feature = "autodiff")]
pub mod autodiff;

#[cfg(test)]
pub(crate) mod test;

use crate::{
    constitutive::ConstitutiveError,
    math::{ContractFirstSecondWithSecond, ContractSecondWithFirst, IDENTITY, Quantity, Rank2},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, CauchyTangentStiffness, DeformationGradient,
        DeformationGradientRate, FirstPiolaKirchhoffRateTangentStiffness,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
        SecondPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::Viscosity,
};
use core::fmt::Debug;

/// Required methods for viscous fluid constitutive models.
pub trait Viscous
where
    Self: Clone + Debug,
{
    /// Returns the bulk viscosity.
    fn bulk_viscosity(&self) -> Quantity<Viscosity>;
    /// Returns the shear viscosity.
    fn shear_viscosity(&self) -> Quantity<Viscosity>;
    /// Calculates and returns the viscous Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
    /// ```
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
    ///
    /// ```math
    /// \mathcal{V}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial\dot{F}_{kL}} = J^{-1}\mathcal{W}_{MNkL}F_{iM}F_{jN}
    /// ```
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
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
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
    ///
    /// ```math
    /// \mathcal{U}_{iJkL} = \frac{\partial P_{iJ}}{\partial\dot{F}_{kL}} = J\mathcal{V}_{iskL}F_{sJ}^{-T}
    /// ```
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
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
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
    ///
    /// ```math
    /// \mathcal{W}_{IJkL} = \frac{\partial S_{IJ}}{\partial\dot{F}_{kL}} = J\mathcal{V}_{mnkL}F_{mI}^{-T}F_{nJ}^{-T}
    /// ```
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
    /// Calculates and returns the tangent stiffness associated with the viscous Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
    /// ```
    fn viscous_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress =
            self.viscous_cauchy_stress(deformation_gradient, deformation_gradient_rate)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .viscous_second_piola_kirchhoff_tangent_stiffness(
                deformation_gradient,
                deformation_gradient_rate,
            )?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the tangent stiffness associated with the viscous first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// ```
    fn viscous_first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress = self.viscous_first_piola_kirchhoff_stress(
            deformation_gradient,
            deformation_gradient_rate,
        )?;
        Ok(self
            .viscous_cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_second_with_first(&deformation_gradient_inverse_transpose)
            * deformation_gradient.determinant()
            + FirstPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - FirstPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            ))
    }
    /// Calculates and returns the tangent stiffness associated with the viscous second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// ```
    fn viscous_second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress = self.viscous_second_piola_kirchhoff_stress(
            deformation_gradient,
            deformation_gradient_rate,
        )?;
        Ok(self
            .viscous_cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_first_second_with_second(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant()
            + SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(
                &deformation_gradient_inverse,
                &second_piola_kirchhoff_stress,
            ))
    }
}
