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

// impl<P, C1, C2> Constitutive<P> for ThermohyperelasticThermalConduction<C1, C2>
// where
//     C1: Constitutive<P>,
//     C2: Constitutive<P>,
// {
//     /// Dummy method that will panic, use [Self::construct()] instead.
//     fn new(_parameters: P) -> Self {
//         panic!()
//     }
// }

impl<P, C1, C2> Solid<P> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<P>,
    C2: ThermalConduction<P>,
{
    fn bulk_modulus(&self) -> &Scalar {
        self.solid_constitutive_model().bulk_modulus()
    }
    fn shear_modulus(&self) -> &Scalar {
        self.solid_constitutive_model().shear_modulus()
    }
}

impl<P, C1, C2> Thermoelastic<P> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<P>,
    C2: ThermalConduction<P>,
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

impl<P, C1, C2> Thermohyperelastic<P> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<P>,
    C2: ThermalConduction<P>,
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

impl<P, C1, C2> Thermal<P> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Constitutive<P>,
    C2: Constitutive<P>,
{
}

impl<P, C1, C2> ThermalConduction<P> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<P>,
    C2: ThermalConduction<P>,
{
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        self.thermal_constitutive_model()
            .heat_flux(temperature_gradient)
    }
}

// impl<P, C1, C2> Multiphysics<P> for ThermohyperelasticThermalConduction<C1, C2>
// where
//     C1: Constitutive<P>,
//     C2: Constitutive<P>,
// {
// }

impl<P, C1, C2> SolidThermal<P, C1, C2> for ThermohyperelasticThermalConduction<C1, C2>
where
    C1: Thermohyperelastic<P>,
    C2: ThermalConduction<P>,
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
