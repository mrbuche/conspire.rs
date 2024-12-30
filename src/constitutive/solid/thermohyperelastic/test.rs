use crate::{constitutive::solid::thermoelastic::test::ALMANSIHAMELPARAMETERS, mechanics::Scalar};
pub const SAINTVENANTKIRCHOFFPARAMETERS: &[Scalar; 4] = &[
    ALMANSIHAMELPARAMETERS[0],
    ALMANSIHAMELPARAMETERS[1],
    ALMANSIHAMELPARAMETERS[2],
    ALMANSIHAMELPARAMETERS[3],
];

macro_rules! calculate_helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed.calculate_helmholtz_free_energy_density(
            $deformation_gradient,
            $constitutive_model_constructed.get_reference_temperature(),
        )
    };
}
pub(crate) use calculate_helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_thermoelastic_macros {
    () => {
        use crate::constitutive::solid::thermoelastic::test::{
            calculate_cauchy_stress_from_deformation_gradient,
            calculate_cauchy_stress_from_deformation_gradient_rotated,
            calculate_cauchy_stress_from_deformation_gradient_simple,
            calculate_cauchy_tangent_stiffness_from_deformation_gradient,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_simple,
            calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient,
            calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient_simple,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_simple,
            calculate_second_piola_kirchoff_tangent_stiffness_from_deformation_gradient,
        };
    };
}
pub(crate) use use_thermoelastic_macros;

macro_rules! test_solid_thermohyperelastic_constitutive_model {
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) => {
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model!(
            $constitutive_model,
            $constitutive_model_parameters,
            $constitutive_model_constructed
        );
        crate::constitutive::solid::thermoelastic::test::test_solid_thermal_constitutive_model!(
            $constitutive_model,
            $constitutive_model_parameters,
            $constitutive_model_constructed
        );
    };
}
pub(crate) use test_solid_thermohyperelastic_constitutive_model;
