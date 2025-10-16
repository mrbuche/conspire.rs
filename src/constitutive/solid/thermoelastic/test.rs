use crate::mechanics::Scalar;

pub use crate::constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS};

pub const COEFFICIENT_OF_THERMAL_EXPANSION: Scalar = 1.0;
pub const REFERENCE_TEMPERATURE: Scalar = 100.0;

macro_rules! cauchy_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient;

macro_rules! cauchy_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress(
            $deformation_gradient,
            $constitutive_model.reference_temperature(),
        )
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_simple;

macro_rules! cauchy_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_rotated;

macro_rules! cauchy_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_tangent_stiffness($deformation_gradient, &get_temperature())
    };
}
pub(crate) use cauchy_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress(
            $deformation_gradient,
            $constitutive_model.reference_temperature(),
        )
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model
            .first_piola_kirchhoff_tangent_stiffness($deformation_gradient, &get_temperature())
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_tangent_stiffness(
            $deformation_gradient,
            &$constitutive_model.reference_temperature(),
        )
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress(
            $deformation_gradient,
            $constitutive_model.reference_temperature(),
        )
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient, &get_temperature())
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model
            .second_piola_kirchhoff_tangent_stiffness($deformation_gradient, &get_temperature())
    };
}
pub(crate) use second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! test_solid_thermal_constitutive_model {
    ($constitutive_model: expr) => {
        use crate::mechanics::test::get_temperature;
        #[test]
        fn coefficient_of_thermal_expansion() -> Result<(), TestError> {
            let model = $constitutive_model;
            let deformation_gradient = DeformationGradient::identity();
            let temperature = model.reference_temperature() - EPSILON;
            let first_piola_kirchhoff_stress =
                model.first_piola_kirchhoff_stress(&deformation_gradient, &temperature)?;
            let compare = 3.0 * model.bulk_modulus() * EPSILON;
            (0..3).try_for_each(|i| {
                (0..3).try_for_each(|j| {
                    if i == j {
                        assert!(
                            (first_piola_kirchhoff_stress[i][j] / compare
                                - model.coefficient_of_thermal_expansion())
                            .abs()
                                < EPSILON
                        );
                        Ok(())
                    } else {
                        assert_eq(&first_piola_kirchhoff_stress[i][j], &0.0)
                    }
                })
            })
        }
    };
}
pub(crate) use test_solid_thermal_constitutive_model;

macro_rules! test_solid_thermoelastic_constitutive_model {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::elastic::test::test_solid_elastic_constitutive_model_no_root!(
            $constitutive_model
        );
        crate::constitutive::solid::thermoelastic::test::test_solid_thermal_constitutive_model!(
            $constitutive_model
        );
    };
}
pub(crate) use test_solid_thermoelastic_constitutive_model;
