use crate::math::{
    Dimensionless, EnergyDensity, Modulus, Quantity, ReciprocalTemperature, Stress, Temperature,
    TensorRank4,
};
#[cfg(test)]
mod test;

use super::*;

/// The Saint Venant-Kirchhoff thermohyperelastic solid constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The coefficient of thermal expansion $`\alpha`$.
/// - The reference temperature $`T_\mathrm{ref}`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The temperature $`T`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Green-Saint Venant strain measure is given by $`\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})`$.
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The coefficient of thermal expansion $`\alpha`$.
    pub coefficient_of_thermal_expansion: Scalar,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Scalar,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl Thermoelastic for SaintVenantKirchhoff {
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}, T) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1} - 3\alpha\kappa(T - T_\mathrm{ref})\mathbf{1}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5)
                .deviatoric_and_trace();
        Ok(
            (deviatoric_strain * (2.0 * Quantity::<Stress>::new(self.shear_modulus()))
                + IDENTITY_00
                    * (Quantity::<Stress>::new(self.bulk_modulus())
                        * (strain_trace
                            - 3.0
                                * Quantity::<ReciprocalTemperature>::new(
                                    self.coefficient_of_thermal_expansion(),
                                )
                                * (Quantity::<Temperature>::new(temperature)
                                    - Quantity::<Temperature>::new(
                                        self.reference_temperature(),
                                    )))))
            .with_unit::<Dimensionless>(),
        )
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\delta_{JL}F_{kI} + \mu\,\delta_{IL}F_{kJ} + \left(\kappa - \frac{2}{3}\,\mu\right)\delta_{IJ}F_{kL}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Scalar,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * Quantity::<Stress>::new(self.shear_modulus());
        Ok(
            (TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00
                        * (Quantity::<Stress>::new(self.bulk_modulus())
                            - TWO_THIRDS * Quantity::<Stress>::new(self.shear_modulus()))),
                    deformation_gradient,
                ))
            .with_unit::<Dimensionless>(),
        )
    }
    fn coefficient_of_thermal_expansion(&self) -> Scalar {
        self.coefficient_of_thermal_expansion
    }
    fn reference_temperature(&self) -> Scalar {
        self.reference_temperature
    }
}

impl Thermohyperelastic for SaintVenantKirchhoff {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}, T) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{1}{2}\left(\kappa - \frac{2}{3}\,\mu\right)\mathrm{tr}(\mathbf{E})^2 - 3\alpha\kappa\,\mathrm{tr}(\mathbf{E})(T - T_\mathrm{ref})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = (deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5;
        let strain_trace = strain.trace();
        Ok(
            (Quantity::<Modulus>::new(self.shear_modulus()) * strain.squared_trace()
                + 0.5
                    * (Quantity::<Modulus>::new(self.bulk_modulus())
                        - TWO_THIRDS * Quantity::<Modulus>::new(self.shear_modulus()))
                    * strain_trace.powi(2)
                - 3.0
                    * Quantity::<Modulus>::new(self.bulk_modulus())
                    * Quantity::<ReciprocalTemperature>::new(
                        self.coefficient_of_thermal_expansion(),
                    )
                    * (Quantity::<Temperature>::new(temperature)
                        - Quantity::<Temperature>::new(self.reference_temperature()))
                    * strain_trace)
                .value_as::<EnergyDensity>(),
        )
    }
}
