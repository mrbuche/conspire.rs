//! Thermoelastic constitutive models.

#[cfg(test)]
pub mod test;

mod almansi_hamel;

pub use almansi_hamel::AlmansiHamel;

use super::*;

/// Required methods for thermoelastic constitutive models.
pub trait Thermoelastic<'a>
where
    Self: Solid<'a>,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
    /// ```
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.calculate_second_piola_kirchoff_stress(deformation_gradient, temperature)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
    /// ```
    fn calculate_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.calculate_cauchy_stress(deformation_gradient, temperature)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .calculate_second_piola_kirchoff_tangent_stiffness(deformation_gradient, temperature)?
            .contract_first_second_indices_with_second_indices_of(
                deformation_gradient,
                deformation_gradient,
            )
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the first Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
    fn calculate_first_piola_kirchoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchoffStress, ConstitutiveError> {
        Ok(
            self.calculate_cauchy_stress(deformation_gradient, temperature)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// ```
    fn calculate_first_piola_kirchoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchoff_stress =
            self.calculate_first_piola_kirchoff_stress(deformation_gradient, temperature)?;
        Ok(self
            .calculate_cauchy_tangent_stiffness(deformation_gradient, temperature)?
            .contract_second_index_with_first_index_of(&deformation_gradient_inverse_transpose)
            * deformation_gradient.determinant()
            + FirstPiolaKirchoffTangentStiffness::dyad_ij_kl(
                &first_piola_kirchoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - FirstPiolaKirchoffTangentStiffness::dyad_il_kj(
                &first_piola_kirchoff_stress,
                &deformation_gradient_inverse_transpose,
            ))
    }
    /// Calculates and returns the second Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn calculate_second_piola_kirchoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.calculate_cauchy_stress(deformation_gradient, temperature)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// ```
    fn calculate_second_piola_kirchoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchoff_stress =
            self.calculate_second_piola_kirchoff_stress(deformation_gradient, temperature)?;
        Ok(self
            .calculate_cauchy_tangent_stiffness(deformation_gradient, temperature)?
            .contract_first_second_indices_with_second_indices_of(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant()
            + SecondPiolaKirchoffTangentStiffness::dyad_ij_kl(
                &second_piola_kirchoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchoffTangentStiffness::dyad_il_kj(
                &second_piola_kirchoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchoffTangentStiffness::dyad_ik_jl(
                &deformation_gradient_inverse,
                &second_piola_kirchoff_stress,
            ))
    }
    /// Returns the coefficient of thermal expansion.
    fn get_coefficient_of_thermal_expansion(&self) -> &Scalar;
    /// Returns the reference temperature.
    fn get_reference_temperature(&self) -> &Scalar;
}
