#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        hybrid::{Additive, Hybrid},
        solid::{elastic::Elastic, Solid},
        Constitutive, ConstitutiveError, Parameters,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Constitutive<'a> for Additive<C1, C2> {
    /// Dummy method that will panic, use [Self::construct()] instead.
    fn new(_parameters: Parameters<'a>) -> Self {
        panic!()
    }
}

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Solid<'a> for Additive<C1, C2> {
    /// Dummy method that will panic.
    fn get_bulk_modulus(&self) -> &Scalar {
        panic!()
    }
    /// Dummy method that will panic.
    fn get_shear_modulus(&self) -> &Scalar {
        panic!()
    }
}

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Elastic<'a> for Additive<C1, C2> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \boldsymbol{\sigma}_1(\mathbf{F}) + \boldsymbol{\sigma}_2(\mathbf{F})
    /// ```
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_cauchy_stress(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_cauchy_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}(\mathbf{F}) = \mathcal{T}_1(\mathbf{F}) + \mathcal{T}_2(\mathbf{F})
    /// ```
    fn calculate_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_cauchy_tangent_stiffness(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_cauchy_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) = \mathbf{P}_1(\mathbf{F}) + \mathbf{P}_2(\mathbf{F})
    /// ```
    fn calculate_first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_first_piola_kirchhoff_stress(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_first_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}(\mathbf{F}) = \mathcal{C}_1(\mathbf{F}) + \mathcal{C}_2(\mathbf{F})
    /// ```
    fn calculate_first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = \mathbf{S}_1(\mathbf{F}) + \mathbf{S}_2(\mathbf{F})
    /// ```
    fn calculate_second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_second_piola_kirchhoff_stress(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_second_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}(\mathbf{F}) = \mathcal{G}_1(\mathbf{F}) + \mathcal{G}_2(\mathbf{F})
    /// ```
    fn calculate_second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self
            .get_constitutive_model_1()
            .calculate_second_piola_kirchhoff_tangent_stiffness(deformation_gradient)?
            + self
                .get_constitutive_model_2()
                .calculate_second_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
}
