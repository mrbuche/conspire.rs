#[cfg(test)]
mod test;

use super::*;
use crate::{
    math::{Quantity, TensorRank4},
    units::{ReciprocalTemperature, Stress, Temperature},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct AlmansiHamel {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The coefficient of thermal expansion $`\alpha`$.
    pub coefficient_of_thermal_expansion: Quantity<ReciprocalTemperature>,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Quantity<Temperature>,
}

impl Solid for AlmansiHamel {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Thermoelastic for AlmansiHamel {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY
                * (self.bulk_modulus() / jacobian
                    * (strain_trace
                        - 3.0
                            * self.coefficient_of_thermal_expansion()
                            * (temperature - self.reference_temperature()))))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let inverse_left_cauchy_green_deformation = &inverse_transpose_deformation_gradient
            * inverse_transpose_deformation_gradient.transpose();
        let strain = (IDENTITY - &inverse_left_cauchy_green_deformation) * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok((TensorRank4::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &inverse_left_cauchy_green_deformation,
        ) + TensorRank4::dyad_ik_jl(
            &inverse_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        )) * (self.shear_modulus() / jacobian)
            + TensorRank4::dyad_ij_kl(
                &IDENTITY,
                &(inverse_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient
                    * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
            )
            - TensorRank4::dyad_ij_kl(
                &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                    + IDENTITY
                        * (self.bulk_modulus() / jacobian
                            * (strain_trace
                                - 3.0
                                    * self.coefficient_of_thermal_expansion()
                                    * (temperature - self.reference_temperature())))),
                &inverse_transpose_deformation_gradient,
            ))
    }
    fn coefficient_of_thermal_expansion(&self) -> Quantity<ReciprocalTemperature> {
        self.coefficient_of_thermal_expansion
    }
    fn reference_temperature(&self) -> Quantity<Temperature> {
        self.reference_temperature
    }
}
