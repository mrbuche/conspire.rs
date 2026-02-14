use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::Plastic,
            viscoplastic::{Viscoplastic, ViscoplasticStateVariables},
        },
        hybrid::Additive,
        solid::{
            Solid, elastic::Elastic, elastic_plastic::ElasticPlasticOrViscoplastic,
            elastic_viscoplastic::ElasticViscoplastic,
        },
    },
    math::Tensor,
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, MandelStressElastic,
        Scalar, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<C1, C2> Solid for Additive<C1, C2>
where
    C1: Solid,
    C2: Elastic,
{
    fn bulk_modulus(&self) -> Scalar {
        self.0.bulk_modulus() + self.1.bulk_modulus()
    }
    fn shear_modulus(&self) -> Scalar {
        self.0.shear_modulus() + self.1.shear_modulus()
    }
}

impl<C1, C2> Plastic for Additive<C1, C2>
where
    C1: Plastic,
    C2: Elastic,
{
    fn initial_yield_stress(&self) -> Scalar {
        self.0.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Scalar {
        self.0.hardening_slope()
    }
}

impl<C1, C2, Y> Viscoplastic<Y> for Additive<C1, C2>
where
    C1: ElasticViscoplastic<Y>,
    C2: Elastic,
    Y: Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<Y> {
        self.0.initial_state()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticStateVariables<Y>, ConstitutiveError> {
        self.0.plastic_evolution(mandel_stress, state_variables)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.0.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.0.reference_flow_rate()
    }
}

impl<C1, C2> ElasticPlasticOrViscoplastic for Additive<C1, C2>
where
    C1: ElasticPlasticOrViscoplastic,
    C2: Elastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \boldsymbol{\sigma}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \boldsymbol{\sigma}_2(\mathbf{F})
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(self
            .0
            .cauchy_stress(deformation_gradient, deformation_gradient_p)?
            + self.1.cauchy_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \mathcal{T}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \mathcal{T}_2(\mathbf{F})
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        Ok(self
            .0
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
            + self.1.cauchy_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \mathbf{P}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \mathbf{P}_2(\mathbf{F})
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .0
            .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            + self.1.first_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \mathcal{C}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \mathcal{C}_2(\mathbf{F})
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self.0.first_piola_kirchhoff_tangent_stiffness(
            deformation_gradient,
            deformation_gradient_p,
        )? + self
            .1
            .first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \mathbf{S}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \mathbf{S}_2(\mathbf{F})
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .0
            .second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            + self.1.second_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}(\mathbf{F},\mathbf{F}_\mathrm{p}) = \mathcal{G}_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + \mathcal{G}_2(\mathbf{F})
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self.0.second_piola_kirchhoff_tangent_stiffness(
            deformation_gradient,
            deformation_gradient_p,
        )? + self
            .1
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
}

impl<C1, C2, Y> ElasticViscoplastic<Y> for Additive<C1, C2>
where
    C1: ElasticViscoplastic<Y>,
    C2: Elastic,
    Y: Tensor,
{
    fn state_variables_evolution(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticStateVariables<Y>, ConstitutiveError> {
        self.0
            .state_variables_evolution(deformation_gradient, state_variables)
    }
}
