use crate::{
    constitutive::{ConstitutiveError, solid::Solid},
    math::Rank2,
    mechanics::{
        CauchyStress, DeformationGradient, DeformationGradientElastic, DeformationGradientPlastic,
        DeformationGradientRatePlastic, FirstPiolaKirchhoffStress, SecondPiolaKirchhoffStress,
        StretchingRatePlastic,
    },
};

/// Required methods for elastic-plastic constitutive models.
pub trait ElasticPlastic
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
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, deformation_gradient_p)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
    }
    /// Calculates and returns the rate plastic deformation.
    ///
    /// ```math
    /// \dot{\mathbf{F}}_\mathrm{p} = \mathbf{D}_\mathrm{p}\cdot\mathbf{F}_\mathrm{p}
    /// ```
    fn plastic_deformation_gradient_rate(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<DeformationGradientRatePlastic, ConstitutiveError> {
        //
        // May be able to pass in deformation_gradient_e directly,
        // since the evaluation of the plastic deformation gradient rate
        // will be preceeded by an inner solve at it will be known at that point.
        // Could be similar for the stress and other evaluations too,
        // since have to write out the all functions in the solves anyway.
        // May be able to satisfy both things if depend on (Fe, Fp) everywhere.
        //
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        Ok(self.plastic_stretching_rate(&deformation_gradient_e)? * deformation_gradient_p)
    }
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = \dot{\gamma}_\mathrm{p}\,\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deformation_gradient_e: &DeformationGradientElastic,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        todo!()
    }
}
