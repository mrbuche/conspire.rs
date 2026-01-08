#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscous::Viscous,
        solid::{
            Solid, TWO_THIRDS, elastic_hyperviscous::ElasticHyperviscous,
            viscoelastic::Viscoelastic,
        },
    },
    math::{IDENTITY, Rank2},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
        Scalar,
    },
};

/// The Almansi-Hamel viscoelastic solid constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The bulk viscosity $`\zeta`$.
/// - The shear viscosity $`\eta`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The deformation gradient rate $`\dot{\mathbf{F}}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Almansi-Hamel strain measure is given by $`\mathbf{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}^{-1})`$.
#[derive(Clone, Debug)]
pub struct AlmansiHamel {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Scalar,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Scalar,
}

impl Solid for AlmansiHamel {
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl Viscous for AlmansiHamel {
    fn bulk_viscosity(&self) -> Scalar {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Scalar {
        self.shear_viscosity
    }
}

impl Viscoelastic for AlmansiHamel {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F},\dot\mathbf{F}) = 2\mu\mathbf{e}' + \kappa\,\mathrm{tr}(\mathbf{e})\mathbf{1} + 2\eta\mathbf{D}' + \zeta\,\mathrm{tr}(\mathbf{D})\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        let velocity_gradient = deformation_gradient_rate * inverse_deformation_gradient;
        let strain_rate = (&velocity_gradient + velocity_gradient.transpose()) * 0.5;
        let (deviatoric_strain_rate, strain_rate_trace) = strain_rate.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + deviatoric_strain_rate * (2.0 * self.shear_viscosity() / jacobian)
            + IDENTITY
                * ((self.bulk_modulus() * strain_trace
                    + self.bulk_viscosity() * strain_rate_trace)
                    / jacobian))
    }
    /// Calculates and returns the rate tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{V}_{IJkL}(\mathbf{F}) = \eta\,\delta_{ik}F_{jL}^{-T} + \eta\,\delta_{jk}F_{iL}^{-T} + \left(\zeta - \frac{2}{3}\,\eta\right)\delta_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let scaled_deformation_gradient_inverse_transpose =
            &deformation_gradient_inverse_transpose * self.shear_viscosity() / jacobian;
        Ok(CauchyRateTangentStiffness::dyad_ik_jl(
            &IDENTITY,
            &scaled_deformation_gradient_inverse_transpose,
        ) + CauchyRateTangentStiffness::dyad_il_jk(
            &scaled_deformation_gradient_inverse_transpose,
            &IDENTITY,
        ) + CauchyRateTangentStiffness::dyad_ij_kl(
            &(IDENTITY
                * ((self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity()) / jacobian)),
            &deformation_gradient_inverse_transpose,
        ))
    }
}

impl ElasticHyperviscous for AlmansiHamel {
    /// Calculates and returns the viscous dissipation.
    ///
    /// ```math
    /// \phi(\mathbf{F},\dot{\mathbf{F}}) = \eta\,\mathrm{tr}(\mathbf{D}^2) + \frac{1}{2}\left(\zeta - \frac{2}{3}\,\eta\right)\mathrm{tr}(\mathbf{D})^2
    /// ```
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let velocity_gradient = deformation_gradient_rate * deformation_gradient.inverse();
        let strain_rate = (&velocity_gradient + velocity_gradient.transpose()) * 0.5;
        Ok(self.shear_viscosity() * strain_rate.squared_trace()
            + 0.5
                * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())
                * strain_rate.trace().powi(2))
    }
}
