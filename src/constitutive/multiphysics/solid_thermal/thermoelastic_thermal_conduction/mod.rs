//! Thermoelastic-thermal conduction constitutive models.

#[cfg(test)]
pub mod test;

use super::*;
use crate::mechanics::{
    CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
    FirstPiolaKirchhoffTangentStiffness, HeatFlux, Scalar, SecondPiolaKirchhoffStress,
    SecondPiolaKirchhoffTangentStiffness, TemperatureGradient,
};

/// A thermoelastic-thermal conduction constitutive model.
pub struct ThermoelasticThermalConduction<C1, C2> {
    thermoelastic_constitutive_model: C1,
    thermal_conduction_constitutive_model: C2,
}

impl<'a, C1, C2> Constitutive<'a> for ThermoelasticThermalConduction<C1, C2> {
    /// Dummy method that will panic, use [Self::construct()] instead.
    fn new(_parameters: Parameters<'a>) -> Self {
        panic!()
    }
}

impl<'a, C1, C2> Solid<'a> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn get_bulk_modulus(&self) -> &Scalar {
        self.get_solid_constitutive_model().get_bulk_modulus()
    }
    fn get_shear_modulus(&self) -> &Scalar {
        self.get_solid_constitutive_model().get_shear_modulus()
    }
}

impl<'a, C1, C2> Thermoelastic<'a> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .cauchy_stress(deformation_gradient, temperature)
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .cauchy_tangent_stiffness(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .first_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .first_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .second_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.get_solid_constitutive_model()
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn get_coefficient_of_thermal_expansion(&self) -> &Scalar {
        self.get_solid_constitutive_model()
            .get_coefficient_of_thermal_expansion()
    }
    fn get_reference_temperature(&self) -> &Scalar {
        self.get_solid_constitutive_model()
            .get_reference_temperature()
    }
}

impl<C1, C2> Thermal<'_> for ThermoelasticThermalConduction<C1, C2> {}

impl<'a, C1, C2> ThermalConduction<'a> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        self.get_thermal_constitutive_model()
            .heat_flux(temperature_gradient)
    }
}

impl<C1, C2> Multiphysics<'_> for ThermoelasticThermalConduction<C1, C2> {}

impl<'a, C1, C2> SolidThermal<'a, C1, C2> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic<'a>,
    C2: ThermalConduction<'a>,
{
    fn construct(
        thermoelastic_constitutive_model: C1,
        thermal_conduction_constitutive_model: C2,
    ) -> Self {
        Self {
            thermoelastic_constitutive_model,
            thermal_conduction_constitutive_model,
        }
    }
    fn get_solid_constitutive_model(&self) -> &C1 {
        &self.thermoelastic_constitutive_model
    }
    fn get_thermal_constitutive_model(&self) -> &C2 {
        &self.thermal_conduction_constitutive_model
    }
}
