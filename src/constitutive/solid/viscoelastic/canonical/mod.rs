#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::{hyperviscous::Hyperviscous, viscous::Viscous},
        solid::{elastic::Elastic, viscoelastic::Viscoelastic},
    },
    math::Quantity,
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        SecondPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffStress,
    },
    units::{Dissipation, Viscosity},
};

impl<C1, C2> Viscous for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscous,
{
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.1.bulk_viscosity()
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.1.shear_viscosity()
    }
    fn viscous_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.1
            .viscous_cauchy_stress(deformation_gradient, deformation_gradient_rate)
    }
    fn viscous_cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        self.1
            .viscous_cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)
    }
    fn viscous_first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.1
            .viscous_first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)
    }
    fn viscous_first_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        self.1.viscous_first_piola_kirchhoff_rate_tangent_stiffness(
            deformation_gradient,
            deformation_gradient_rate,
        )
    }
    fn viscous_second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.1
            .viscous_second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)
    }
    fn viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        self.1
            .viscous_second_piola_kirchhoff_rate_tangent_stiffness(
                deformation_gradient,
                deformation_gradient_rate,
            )
    }
}

impl<C1, C2> Viscoelastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscous,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(self.0.cauchy_stress(deformation_gradient)?
            + self
                .1
                .viscous_cauchy_stress(deformation_gradient, deformation_gradient_rate)?)
    }
    fn cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        self.1
            .viscous_cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)
    }
}

impl<C1, C2> Hyperviscous for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Hyperviscous,
{
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)
    }
}
