use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::{Plastic, StateVariables},
            viscoplastic::Viscoplastic,
        },
        hybrid::Additive,
        solid::{
            Solid, elastic::Elastic, elastic_plastic::ElasticPlasticOrViscoplastic,
            elastic_viscoplastic::ElasticViscoplastic,
        },
    },
    math::Rank2,
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<C1, C2> Solid for Additive<C1, C2>
where
    C1: ElasticViscoplastic,
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
    C1: ElasticViscoplastic,
    C2: Elastic,
{
    fn initial_yield_stress(&self) -> Scalar {
        self.0.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Scalar {
        self.0.hardening_slope()
    }
}

impl<C1, C2> Viscoplastic for Additive<C1, C2>
where
    C1: ElasticViscoplastic,
    C2: Elastic,
{
    fn rate_sensitivity(&self) -> Scalar {
        self.0.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.0.reference_flow_rate()
    }
}

impl<C1, C2> ElasticPlasticOrViscoplastic for Additive<C1, C2>
where
    C1: ElasticViscoplastic,
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

impl<C1, C2> ElasticViscoplastic for Additive<C1, C2>
where
    C1: ElasticViscoplastic,
    C2: Elastic,
{
    fn state_variables_evolution(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &StateVariables,
    ) -> Result<StateVariables, ConstitutiveError> {
        let (deformation_gradient_p, _) = state_variables.into();
        let jacobian = self.0.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self
            .0
            .cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_e = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        self.0.plastic_evolution(mandel_stress_e, state_variables)
    }
}
