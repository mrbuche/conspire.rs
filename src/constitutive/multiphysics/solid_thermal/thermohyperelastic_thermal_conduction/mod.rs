//! Thermohyperelastic-thermal conduction constitutive models.

#[cfg(test)]
mod test;

use super::*;
use crate::mechanics::{
    CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
    FirstPiolaKirchhoffTangentStiffness, HeatFlux, Scalar, SecondPiolaKirchhoffStress,
    SecondPiolaKirchhoffTangentStiffness, TemperatureGradient,
};

/// A thermohyperelastic-thermal conduction constitutive model.
#[derive(Debug)]
pub struct ThermohyperelasticThermalConduction<C1, C2> {
    thermohyperelastic_constitutive_model: C1,
    thermal_conduction_constitutive_model: C2,
}

impl<'a, C1, C2> Constitutive<'a> for ThermohyperelasticThermalConduction<C1, C2> {
    /// Dummy method that will panic, use [Self::construct()] instead.
    fn new(_parameters: Parameters<'a>) -> Self {
        panic!()
    }
}

impl<'a, C1, C2> Solid<'a> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn bulk_modulus(&self) -> &Scalar {
        self.solid_constitutive_model().bulk_modulus()
    }
    fn shear_modulus(&self) -> &Scalar {
        self.solid_constitutive_model().shear_modulus()
    }
}

impl<'a, C1, C2> Thermoelastic<'a> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .cauchy_stress(deformation_gradient, temperature)
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .cauchy_tangent_stiffness(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .first_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .first_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .second_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn coefficient_of_thermal_expansion(&self) -> &Scalar {
        self.solid_constitutive_model()
            .coefficient_of_thermal_expansion()
    }
    fn reference_temperature(&self) -> &Scalar {
        self.solid_constitutive_model().reference_temperature()
    }
}

impl<'a, C1, C2> Thermohyperelastic<'a> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<Scalar, ConstitutiveError> {
        self.solid_constitutive_model()
            .helmholtz_free_energy_density(deformation_gradient, temperature)
    }
}

impl<C1, C2> Thermal<'_> for ThermohyperelasticThermalConduction<C1, C2> {}

impl<'a, C1, C2> ThermalConduction<'a> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        self.thermal_constitutive_model()
            .heat_flux(temperature_gradient)
    }
}

impl<C1, C2> Multiphysics<'_> for ThermohyperelasticThermalConduction<C1, C2> {}

impl<'a, C1, C2> SolidThermal<'a, C1, C2> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn construct(
        thermohyperelastic_constitutive_model: C1,
        thermal_conduction_constitutive_model: C2,
    ) -> Self {
        Self {
            thermohyperelastic_constitutive_model,
            thermal_conduction_constitutive_model,
        }
    }
    fn solid_constitutive_model(&self) -> &C1 {
        &self.thermohyperelastic_constitutive_model
    }
    fn thermal_constitutive_model(&self) -> &C2 {
        &self.thermal_conduction_constitutive_model
    }
}
