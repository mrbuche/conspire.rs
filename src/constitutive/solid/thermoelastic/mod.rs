//! Thermoelastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

use super::*;
use crate::{
    math::Quantity,
    mechanics::{
        CauchyMixedTangentStiffness, CauchyThermalTangent, CauchyThermalTangentStiffness,
        FirstPiolaKirchhoffMixedTangentStiffness, FirstPiolaKirchhoffThermalTangent,
        FirstPiolaKirchhoffThermalTangentStiffness, SecondPiolaKirchhoffMixedTangentStiffness,
        SecondPiolaKirchhoffThermalTangent, SecondPiolaKirchhoffThermalTangentStiffness,
    },
    units::Temperature,
};

mod almansi_hamel;

pub use almansi_hamel::AlmansiHamel;

/// Required methods for thermoelastic solid constitutive models.
pub trait Thermoelastic
where
    Self: Solid,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, temperature)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, temperature)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self.cauchy_stress(deformation_gradient, temperature)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, temperature)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, temperature)?
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
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.cauchy_stress(deformation_gradient, temperature)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress =
            self.second_piola_kirchhoff_stress(deformation_gradient, temperature)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, temperature)?
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
    /// Calculates and returns the derivative of the Cauchy stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial\boldsymbol{\sigma}}{\partial T} = J^{-1}\mathbf{F}\cdot\frac{\partial\mathbf{S}}{\partial T}\cdot\mathbf{F}^T
    /// ```
    fn cauchy_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangent, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_thermal_tangent(deformation_gradient, temperature)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the derivative of the first Piola-Kirchhoff stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial\mathbf{P}}{\partial T} = J\,\frac{\partial\boldsymbol{\sigma}}{\partial T}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<FirstPiolaKirchhoffThermalTangent, ConstitutiveError> {
        Ok(
            self.cauchy_thermal_tangent(deformation_gradient, temperature)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the derivative of the second Piola-Kirchhoff stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial\mathbf{S}}{\partial T} = \mathbf{F}^{-1}\cdot\frac{\partial\mathbf{P}}{\partial T}
    /// ```
    fn second_piola_kirchhoff_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffThermalTangent, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.cauchy_thermal_tangent(deformation_gradient, temperature)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the derivative of the Cauchy stress with respect to the deformation gradient and temperature.
    ///
    /// ```math
    /// \mathcal{M}_{ijkL} = \frac{\partial^2\sigma_{ij}}{\partial F_{kL}\partial T}
    /// ```
    fn cauchy_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyMixedTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_thermal_tangent =
            self.cauchy_thermal_tangent(deformation_gradient, temperature)?;
        let some_stress = &cauchy_thermal_tangent * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_mixed_tangent_stiffness(deformation_gradient, temperature)?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant()
            - CauchyMixedTangentStiffness::dyad_ij_kl(
                &cauchy_thermal_tangent,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyMixedTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyMixedTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the second derivative of the Cauchy stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial^2\boldsymbol{\sigma}}{\partial T^2} = J^{-1}\mathbf{F}\cdot\frac{\partial^2\mathbf{S}}{\partial T^2}\cdot\mathbf{F}^T
    /// ```
    fn cauchy_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangentStiffness, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_thermal_tangent_stiffness(
                deformation_gradient,
                temperature,
            )?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the derivative of the first Piola-Kirchhoff stress with respect to the deformation gradient and temperature.
    ///
    /// ```math
    /// \frac{\partial^2 P_{iJ}}{\partial F_{kL}\partial T} = J \mathcal{M}_{iskL} F_{sJ}^{-T} + \frac{\partial P_{iJ}}{\partial T} F_{kL}^{-T} - \frac{\partial P_{iL}}{\partial T} F_{kJ}^{-T}
    /// ```
    fn first_piola_kirchhoff_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<FirstPiolaKirchhoffMixedTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_thermal_tangent =
            self.first_piola_kirchhoff_thermal_tangent(deformation_gradient, temperature)?;
        Ok(self
            .cauchy_mixed_tangent_stiffness(deformation_gradient, temperature)?
            .contract_second_with_first(&deformation_gradient_inverse_transpose)
            * deformation_gradient.determinant()
            + FirstPiolaKirchhoffMixedTangentStiffness::dyad_ij_kl(
                &first_piola_kirchhoff_thermal_tangent,
                &deformation_gradient_inverse_transpose,
            )
            - FirstPiolaKirchhoffMixedTangentStiffness::dyad_il_kj(
                &first_piola_kirchhoff_thermal_tangent,
                &deformation_gradient_inverse_transpose,
            ))
    }
    /// Calculates and returns the second derivative of the first Piola-Kirchhoff stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial^2\mathbf{P}}{\partial T^2} = J\,\frac{\partial^2\boldsymbol{\sigma}}{\partial T^2}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<FirstPiolaKirchhoffThermalTangentStiffness, ConstitutiveError> {
        Ok(
            self.cauchy_thermal_tangent_stiffness(deformation_gradient, temperature)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the derivative of the second Piola-Kirchhoff stress with respect to the deformation gradient and temperature.
    ///
    /// ```math
    /// \frac{\partial^2 S_{IJ}}{\partial F_{kL}\partial T} = J \mathcal{M}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + \frac{\partial S_{IJ}}{\partial T} F_{kL}^{-T} - \frac{\partial S_{IL}}{\partial T} F_{kJ}^{-T} - \frac{\partial S_{LJ}}{\partial T} F_{kI}^{-T}
    /// ```
    fn second_piola_kirchhoff_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffMixedTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_thermal_tangent =
            self.second_piola_kirchhoff_thermal_tangent(deformation_gradient, temperature)?;
        Ok(self
            .cauchy_mixed_tangent_stiffness(deformation_gradient, temperature)?
            .contract_first_second_with_second(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant()
            + SecondPiolaKirchhoffMixedTangentStiffness::dyad_ij_kl(
                &second_piola_kirchhoff_thermal_tangent,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffMixedTangentStiffness::dyad_il_kj(
                &second_piola_kirchhoff_thermal_tangent,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffMixedTangentStiffness::dyad_ik_jl(
                &deformation_gradient_inverse,
                &second_piola_kirchhoff_thermal_tangent,
            ))
    }
    /// Calculates and returns the second derivative of the second Piola-Kirchhoff stress with respect to temperature.
    ///
    /// ```math
    /// \frac{\partial^2\mathbf{S}}{\partial T^2} = J\,\mathbf{F}^{-1}\cdot\frac{\partial^2\boldsymbol{\sigma}}{\partial T^2}\cdot\mathbf{F}^{-T}
    /// ```
    fn second_piola_kirchhoff_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffThermalTangentStiffness, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.cauchy_thermal_tangent_stiffness(deformation_gradient, temperature)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Returns the reference temperature.
    fn reference_temperature(&self) -> Quantity<Temperature>;
}
