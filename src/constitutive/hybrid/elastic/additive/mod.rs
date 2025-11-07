#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::{Additive, Hybrid},
        solid::{Solid, elastic::Elastic},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<C1, C2> Solid for Additive<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    fn bulk_modulus(&self) -> Scalar {
        self.constitutive_model_1().bulk_modulus() + self.constitutive_model_2().bulk_modulus()
    }
    fn shear_modulus(&self) -> Scalar {
        self.constitutive_model_1().shear_modulus() + self.constitutive_model_2().shear_modulus()
    }
}

impl<C1, C2> Elastic for Additive<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \boldsymbol{\sigma}_1(\mathbf{F}) + \boldsymbol{\sigma}_2(\mathbf{F})
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .cauchy_stress(deformation_gradient)?
            + self
                .constitutive_model_2()
                .cauchy_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}(\mathbf{F}) = \mathcal{T}_1(\mathbf{F}) + \mathcal{T}_2(\mathbf{F})
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .cauchy_tangent_stiffness(deformation_gradient)?
            + self
                .constitutive_model_2()
                .cauchy_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) = \mathbf{P}_1(\mathbf{F}) + \mathbf{P}_2(\mathbf{F})
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .first_piola_kirchhoff_stress(deformation_gradient)?
            + self
                .constitutive_model_2()
                .first_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}(\mathbf{F}) = \mathcal{C}_1(\mathbf{F}) + \mathcal{C}_2(\mathbf{F})
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?
            + self
                .constitutive_model_2()
                .first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = \mathbf{S}_1(\mathbf{F}) + \mathbf{S}_2(\mathbf{F})
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .second_piola_kirchhoff_stress(deformation_gradient)?
            + self
                .constitutive_model_2()
                .second_piola_kirchhoff_stress(deformation_gradient)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}(\mathbf{F}) = \mathcal{G}_1(\mathbf{F}) + \mathcal{G}_2(\mathbf{F})
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient)?
            + self
                .constitutive_model_2()
                .second_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
    }
}
