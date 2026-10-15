#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::solid::{FIVE_THIRDS, TWO_THIRDS},
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{
        CauchyMixedTangentStiffness, CauchyThermalTangent, CauchyThermalTangentStiffness,
        Deformation,
    },
    units::{
        Dimensionless, EnergyDensity, ReciprocalTemperature, ReciprocalTemperatureSquared, Stress,
        Temperature,
    },
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct NeoHookean {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$ at the reference temperature.
    pub shear_modulus: Quantity<Stress>,
    /// The coefficient of thermal expansion $`\alpha`$ at the reference temperature.
    pub coefficient_of_thermal_expansion: Quantity<ReciprocalTemperature>,
    /// The change of the coefficient of thermal expansion with temperature $`\alpha'`$.
    pub coefficient_of_thermal_expansion_slope: Quantity<ReciprocalTemperatureSquared>,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Quantity<Temperature>,
}

impl Solid for NeoHookean {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl NeoHookean {
    fn shear_modulus_at(&self, temperature: Quantity<Temperature>) -> Quantity<Stress> {
        self.shear_modulus() * (temperature / self.reference_temperature())
    }
    fn thermal_expansion_at(
        &self,
        temperature: Quantity<Temperature>,
    ) -> Quantity<ReciprocalTemperature> {
        self.coefficient_of_thermal_expansion
            + self.coefficient_of_thermal_expansion_slope
                * (temperature - self.reference_temperature())
    }
    fn thermal_strain_at(&self, temperature: Quantity<Temperature>) -> Quantity<Dimensionless> {
        (self.coefficient_of_thermal_expansion
            + self.coefficient_of_thermal_expansion_slope
                * (temperature - self.reference_temperature())
                * 0.5)
            * (temperature - self.reference_temperature())
    }
}

impl Thermoelastic for NeoHookean {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * self.shear_modulus_at(temperature)
                + IDENTITY
                    * (self.bulk_modulus()
                        * (0.5 * (jacobian - 1.0 / jacobian)
                            - 3.0 * self.thermal_strain_at(temperature))),
        )
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_shear_modulus = self.shear_modulus_at(temperature) / jacobian.powf(FIVE_THIRDS);
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))
                    - deformation_gradient.left_cauchy_green().deviatoric()
                        * (scaled_shear_modulus * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            ))
    }
    #[doc = include_str!("cauchy_thermal_tangent.md")]
    fn cauchy_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangent, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * (self.shear_modulus() / self.reference_temperature())
                - IDENTITY * (3.0 * self.bulk_modulus() * self.thermal_expansion_at(temperature)),
        )
    }
    #[doc = include_str!("cauchy_mixed_tangent_stiffness.md")]
    fn cauchy_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyMixedTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scaled_shear_modulus_rate =
            self.shear_modulus() / self.reference_temperature() / jacobian.powf(FIVE_THIRDS);
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus_rate
            - TensorRank4::dyad_ij_kl(
                &(deformation_gradient.left_cauchy_green().deviatoric()
                    * (scaled_shear_modulus_rate * FIVE_THIRDS)),
                &deformation_gradient.inverse_transpose(),
            ))
    }
    #[doc = include_str!("cauchy_thermal_tangent_stiffness.md")]
    fn cauchy_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        Ok(IDENTITY * (-3.0 * self.bulk_modulus() * self.coefficient_of_thermal_expansion_slope))
    }
    fn reference_temperature(&self) -> Quantity<Temperature> {
        self.reference_temperature
    }
}

impl Thermohyperelastic for NeoHookean {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(0.5
            * (self.shear_modulus_at(temperature)
                * (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
                    - 3.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
            - self.bulk_modulus() * (3.0 * self.thermal_strain_at(temperature)) * (jacobian - 1.0))
    }
}
