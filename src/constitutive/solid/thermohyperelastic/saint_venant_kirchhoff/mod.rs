#[cfg(test)]
mod test;
use crate::math::{Quantity, TensorRank4};
use crate::units::{EnergyDensity, ReciprocalTemperature, Stress, Temperature};

use super::*;

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The coefficient of thermal expansion $`\alpha`$.
    pub coefficient_of_thermal_expansion: Quantity<ReciprocalTemperature>,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Quantity<Temperature>,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Thermoelastic for SaintVenantKirchhoff {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5)
                .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00
                * (self.bulk_modulus()
                    * (strain_trace
                        - 3.0
                            * self.coefficient_of_thermal_expansion()
                            * (temperature - self.reference_temperature()))))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * self.shear_modulus();
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                    deformation_gradient,
                ),
        )
    }
    fn coefficient_of_thermal_expansion(&self) -> Quantity<ReciprocalTemperature> {
        self.coefficient_of_thermal_expansion
    }
    fn reference_temperature(&self) -> Quantity<Temperature> {
        self.reference_temperature
    }
}

impl Thermohyperelastic for SaintVenantKirchhoff {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = (deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5;
        let strain_trace = strain.trace();
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain_trace.powi(2)
            - 3.0
                * self.bulk_modulus()
                * self.coefficient_of_thermal_expansion()
                * (temperature - self.reference_temperature())
                * strain_trace)
    }
}
