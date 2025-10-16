pub use crate::constitutive::solid::thermoelastic::test::{
    BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE, SHEAR_MODULUS,
};

macro_rules! helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed.helmholtz_free_energy_density(
            $deformation_gradient,
            $constitutive_model_constructed.reference_temperature(),
        )
    };
}
pub(crate) use helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_thermoelastic_macros {
    () => {
        use crate::constitutive::solid::thermoelastic::test::{
            cauchy_stress_from_deformation_gradient,
            cauchy_stress_from_deformation_gradient_rotated,
            cauchy_stress_from_deformation_gradient_simple,
            cauchy_tangent_stiffness_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            first_piola_kirchhoff_stress_from_deformation_gradient_simple,
            first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient,
            first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple,
            second_piola_kirchhoff_stress_from_deformation_gradient,
            second_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            second_piola_kirchhoff_stress_from_deformation_gradient_simple,
            second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient,
        };
    };
}
pub(crate) use use_thermoelastic_macros;

macro_rules! test_solid_thermohyperelastic_constitutive_model {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_minimize!(
            $constitutive_model
        );
        crate::constitutive::solid::thermoelastic::test::test_solid_thermal_constitutive_model!(
            $constitutive_model
        );
    };
}
pub(crate) use test_solid_thermohyperelastic_constitutive_model;
