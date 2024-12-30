use super::*;
use crate::constitutive::solid::thermoelastic::{test::ALMANSIHAMELPARAMETERS, AlmansiHamel};

test_thermoelastic_thermal_conduction_constitutive_model!(
    ThermoelasticThermalConduction,
    AlmansiHamel,
    ALMANSIHAMELPARAMETERS,
    Fourier,
    FOURIERPARAMETERS
);

macro_rules! test_thermoelastic_thermal_conduction_constitutive_model {
    ($thermoelastic_thermal_conduction_constitutive_model: ident,
     $thermoelastic_constitutive_model: ident, $thermoelastic_constitutive_model_parameters: expr,
     $thermal_conduction_constitutive_model: ident, $thermal_conduction_constitutive_model_parameters: expr) => {
        use crate::{
            constitutive::thermal::conduction::{test::FOURIERPARAMETERS, Fourier},
            mechanics::test::{
                get_deformation_gradient, get_temperature, get_temperature_gradient,
            },
        };
        fn get_thermoelastic_constitutive_model<'a>() -> $thermoelastic_constitutive_model<'a> {
            $thermoelastic_constitutive_model::new($thermoelastic_constitutive_model_parameters)
        }
        fn get_thermal_conduction_constitutive_model<'a>(
        ) -> $thermal_conduction_constitutive_model<'a> {
            $thermal_conduction_constitutive_model::new(
                $thermal_conduction_constitutive_model_parameters,
            )
        }
        fn get_thermoelastic_thermal_conduction_constitutive_model<'a>(
        ) -> $thermoelastic_thermal_conduction_constitutive_model<
            $thermoelastic_constitutive_model<'a>,
            $thermal_conduction_constitutive_model<'a>,
        > {
            $thermoelastic_thermal_conduction_constitutive_model::construct(
                get_thermoelastic_constitutive_model(),
                get_thermal_conduction_constitutive_model(),
            )
        }
        #[test]
        #[should_panic]
        fn new() {
            $thermoelastic_thermal_conduction_constitutive_model::<
                $thermoelastic_constitutive_model,
                $thermal_conduction_constitutive_model,
            >::new(FOURIERPARAMETERS);
        }
        #[test]
        fn get_bulk_modulus() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model().get_bulk_modulus(),
                get_thermoelastic_constitutive_model().get_bulk_modulus()
            )
        }
        #[test]
        fn get_shear_modulus() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model().get_shear_modulus(),
                get_thermoelastic_constitutive_model().get_shear_modulus()
            )
        }
        #[test]
        fn get_coefficient_of_thermal_expansion() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model()
                    .get_coefficient_of_thermal_expansion(),
                get_thermoelastic_constitutive_model().get_coefficient_of_thermal_expansion()
            )
        }
        #[test]
        fn get_reference_temperature() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model()
                    .get_reference_temperature(),
                get_thermoelastic_constitutive_model().get_reference_temperature()
            )
        }
        #[test]
        fn calculate_cauchy_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_cauchy_stress(&get_deformation_gradient(), &get_temperature())?,
                &get_thermoelastic_constitutive_model()
                    .calculate_cauchy_stress(&get_deformation_gradient(), &get_temperature())?,
            )
        }
        #[test]
        fn calculate_cauchy_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_cauchy_tangent_stiffness(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().calculate_cauchy_tangent_stiffness(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn calculate_first_piola_kirchoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_first_piola_kirchoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().calculate_first_piola_kirchoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn calculate_first_piola_kirchoff_tangent_stiffness(
        ) -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_first_piola_kirchoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().calculate_first_piola_kirchoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn calculate_heat_flux() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_heat_flux(&get_temperature_gradient()),
                &get_thermal_conduction_constitutive_model()
                    .calculate_heat_flux(&get_temperature_gradient()),
            )
        }
        #[test]
        fn calculate_second_piola_kirchoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_second_piola_kirchoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().calculate_second_piola_kirchoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn calculate_second_piola_kirchoff_tangent_stiffness(
        ) -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .calculate_second_piola_kirchoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().calculate_second_piola_kirchoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn size() {
            assert_eq!(
                std::mem::size_of::<
                    ThermoelasticThermalConduction<
                        $thermoelastic_constitutive_model,
                        $thermal_conduction_constitutive_model,
                    >,
                >(),
                2 * std::mem::size_of::<crate::constitutive::Parameters>()
            )
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_constitutive_model;
