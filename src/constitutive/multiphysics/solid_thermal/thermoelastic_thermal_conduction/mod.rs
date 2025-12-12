//! Thermoelastic-thermal conduction constitutive models.

#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        multiphysics::{Multiphysics, solid_thermal::SolidThermal},
        solid::{Solid, thermoelastic::Thermoelastic},
        thermal::{Thermal, conduction::ThermalConduction},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, HeatFlux, HeatFluxTangent, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness, TemperatureGradient,
    },
};

/// A thermoelastic-thermal conduction constitutive model.
#[derive(Clone, Debug)]
pub struct ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
    thermoelastic_constitutive_model: C1,
    thermal_conduction_constitutive_model: C2,
}

impl<C1, C2> From<(C1, C2)> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
    fn from(
        (thermoelastic_constitutive_model, thermal_conduction_constitutive_model): (C1, C2),
    ) -> Self {
        Self {
            thermoelastic_constitutive_model,
            thermal_conduction_constitutive_model,
        }
    }
}

impl<C1, C2> Solid for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
    fn bulk_modulus(&self) -> Scalar {
        self.solid_constitutive_model().bulk_modulus()
    }
    fn shear_modulus(&self) -> Scalar {
        self.solid_constitutive_model().shear_modulus()
    }
}

impl<C1, C2> Thermoelastic for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .cauchy_stress(deformation_gradient, temperature)
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .cauchy_tangent_stiffness(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .first_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .first_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        self.solid_constitutive_model()
            .second_piola_kirchhoff_stress(deformation_gradient, temperature)
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Scalar,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        self.solid_constitutive_model()
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, temperature)
    }
    fn coefficient_of_thermal_expansion(&self) -> Scalar {
        self.solid_constitutive_model()
            .coefficient_of_thermal_expansion()
    }
    fn reference_temperature(&self) -> Scalar {
        self.solid_constitutive_model().reference_temperature()
    }
}

impl<C1, C2> Thermal for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
}

impl<C1, C2> ThermalConduction for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        self.thermal_constitutive_model()
            .potential(temperature_gradient)
    }
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        self.thermal_constitutive_model()
            .heat_flux(temperature_gradient)
    }
    fn heat_flux_tangent(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        self.thermal_constitutive_model()
            .heat_flux_tangent(temperature_gradient)
    }
}

impl<C1, C2> Multiphysics for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
    Self: SolidThermal<C1, C2>,
{
}

impl<C1, C2> SolidThermal<C1, C2> for ThermoelasticThermalConduction<C1, C2>
where
    C1: Thermoelastic,
    C2: ThermalConduction,
{
    fn solid_constitutive_model(&self) -> &C1 {
        &self.thermoelastic_constitutive_model
    }
    fn thermal_constitutive_model(&self) -> &C2 {
        &self.thermal_conduction_constitutive_model
    }
}
