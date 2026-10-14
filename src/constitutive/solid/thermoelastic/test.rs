use crate::{
    math::Quantity,
    units::{ReciprocalTemperature, Temperature},
};

pub use crate::constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS};

pub const COEFFICIENT_OF_THERMAL_EXPANSION: Quantity<ReciprocalTemperature> =
    ReciprocalTemperature::per_kelvin(1.0);
pub const REFERENCE_TEMPERATURE: Quantity<Temperature> = Temperature::kelvin(100.0);

macro_rules! cauchy_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient, get_temperature())
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
        $constitutive_model.cauchy_stress($deformation_gradient, get_temperature())
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_rotated;

macro_rules! cauchy_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_tangent_stiffness($deformation_gradient, get_temperature())
    };
}
pub(crate) use cauchy_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient, get_temperature())
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
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient, get_temperature())
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model
            .first_piola_kirchhoff_tangent_stiffness($deformation_gradient, get_temperature())
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_tangent_stiffness(
            $deformation_gradient,
            $constitutive_model.reference_temperature(),
        )
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient, get_temperature())
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
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient, get_temperature())
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model
            .second_piola_kirchhoff_tangent_stiffness($deformation_gradient, get_temperature())
    };
}
pub(crate) use second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! thermal_tangent_finite_difference {
    ($constitutive_model: expr, $stress: ident, $tangent: ident) => {
        #[test]
        fn $tangent() -> Result<(), $crate::math::assert::AssertionError> {
            let model = $constitutive_model;
            let deformation_gradient = $crate::mechanics::test::get_deformation_gradient();
            let temperature = $crate::mechanics::test::get_temperature();
            let step = $crate::math::Quantity::new($crate::EPSILON);
            let finite_difference = (model
                .$stress(&deformation_gradient, temperature + step * 0.5)?
                - model.$stress(&deformation_gradient, temperature - step * 0.5)?)
                / step;
            $crate::math::assert::Assert::default().eq_within_fd_tol(
                &finite_difference,
                &model.$tangent(&deformation_gradient, temperature)?,
            )
        }
    };
}

macro_rules! mixed_tangent_finite_difference {
    ($constitutive_model: expr, $thermal_tangent: ident, $mixed_tangent: ident) => {
        #[test]
        fn $mixed_tangent() -> Result<(), $crate::math::assert::AssertionError> {
            let model = $constitutive_model;
            let deformation_gradient = $crate::mechanics::test::get_deformation_gradient();
            let temperature = $crate::mechanics::test::get_temperature();
            let mixed = model.$mixed_tangent(&deformation_gradient, temperature)?;
            (0..3).try_for_each(|k| {
                (0..3).try_for_each(|l| {
                    let mut plus = deformation_gradient.clone();
                    plus[k][l] += $crate::math::assert::perturbation(0.5 * $crate::EPSILON);
                    let mut minus = deformation_gradient.clone();
                    minus[k][l] -= $crate::math::assert::perturbation(0.5 * $crate::EPSILON);
                    let difference = model.$thermal_tangent(&plus, temperature)?
                        - model.$thermal_tangent(&minus, temperature)?;
                    (0..3).try_for_each(|i| {
                        (0..3).try_for_each(|j| {
                            $crate::math::assert::Assert::default().eq_within_fd_tol(
                                &(difference[i][j] / $crate::EPSILON),
                                &mixed[i][j][k][l],
                            )
                        })
                    })
                })
            })
        }
    };
}

macro_rules! test_solid_thermal_constitutive_model {
    ($constitutive_model: expr) => {
        use crate::mechanics::test::get_temperature;
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            cauchy_stress,
            cauchy_thermal_tangent
        );
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            first_piola_kirchhoff_stress,
            first_piola_kirchhoff_thermal_tangent
        );
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            second_piola_kirchhoff_stress,
            second_piola_kirchhoff_thermal_tangent
        );
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            cauchy_thermal_tangent,
            cauchy_thermal_tangent_stiffness
        );
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            first_piola_kirchhoff_thermal_tangent,
            first_piola_kirchhoff_thermal_tangent_stiffness
        );
        crate::constitutive::solid::thermoelastic::test::thermal_tangent_finite_difference!(
            $constitutive_model,
            second_piola_kirchhoff_thermal_tangent,
            second_piola_kirchhoff_thermal_tangent_stiffness
        );
        crate::constitutive::solid::thermoelastic::test::mixed_tangent_finite_difference!(
            $constitutive_model,
            cauchy_thermal_tangent,
            cauchy_mixed_tangent_stiffness
        );
        crate::constitutive::solid::thermoelastic::test::mixed_tangent_finite_difference!(
            $constitutive_model,
            first_piola_kirchhoff_thermal_tangent,
            first_piola_kirchhoff_mixed_tangent_stiffness
        );
        crate::constitutive::solid::thermoelastic::test::mixed_tangent_finite_difference!(
            $constitutive_model,
            second_piola_kirchhoff_thermal_tangent,
            second_piola_kirchhoff_mixed_tangent_stiffness
        );
        #[test]
        fn coefficient_of_thermal_expansion() -> Result<(), AssertionError> {
            let model = $constitutive_model;
            let deformation_gradient = DeformationGradient::identity();
            let temperature = model.reference_temperature() - $crate::math::Quantity::new(EPSILON);
            let first_piola_kirchhoff_stress =
                model.first_piola_kirchhoff_stress(&deformation_gradient, temperature)?;
            let compare = 3.0 * model.bulk_modulus().value() * EPSILON;
            (0..3).try_for_each(|i| {
                (0..3).try_for_each(|j| {
                    if i == j {
                        assert!(
                            ((first_piola_kirchhoff_stress[i][j] / compare).value()
                                - model.coefficient_of_thermal_expansion().value())
                            .abs()
                                < EPSILON
                        );
                        Ok(())
                    } else {
                        Assert::zero(&first_piola_kirchhoff_stress[i][j])
                    }
                })
            })
        }
    };
}
pub(crate) use mixed_tangent_finite_difference;
pub(crate) use test_solid_thermal_constitutive_model;
pub(crate) use thermal_tangent_finite_difference;

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
