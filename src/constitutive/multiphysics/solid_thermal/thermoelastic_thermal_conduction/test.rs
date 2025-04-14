use super::*;
use crate::constitutive::solid::thermoelastic::{AlmansiHamel, test::ALMANSIHAMELPARAMETERS};

type AlmansiHamelType<'a> = AlmansiHamel<&'a [Scalar; 4]>;
type FourierType<'a> = Fourier<&'a [Scalar; 1]>;

test_thermoelastic_thermal_conduction_constitutive_model!(
    ThermoelasticThermalConduction,
    AlmansiHamelType,
    ALMANSIHAMELPARAMETERS,
    FourierType,
    FOURIERPARAMETERS
);

macro_rules! test_thermoelastic_thermal_conduction_constitutive_model {
    ($thermoelastic_thermal_conduction_constitutive_model: ident,
     $thermoelastic_constitutive_model: ident, $thermoelastic_constitutive_model_parameters: expr,
     $thermal_conduction_constitutive_model: ident, $thermal_conduction_constitutive_model_parameters: expr) => {
        use crate::{
            constitutive::thermal::conduction::{Fourier, test::FOURIERPARAMETERS},
            mechanics::test::{
                get_deformation_gradient, get_temperature, get_temperature_gradient,
            },
        };
        fn get_thermoelastic_constitutive_model<'a>() -> $thermoelastic_constitutive_model<'a> {
            $thermoelastic_constitutive_model::new($thermoelastic_constitutive_model_parameters)
        }
        fn get_thermal_conduction_constitutive_model<'a>()
        -> $thermal_conduction_constitutive_model<'a> {
            $thermal_conduction_constitutive_model::new(
                $thermal_conduction_constitutive_model_parameters,
            )
        }
        fn get_thermoelastic_thermal_conduction_constitutive_model<'a>()
        -> $thermoelastic_thermal_conduction_constitutive_model<
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
        fn bulk_modulus() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model().bulk_modulus(),
                get_thermoelastic_constitutive_model().bulk_modulus()
            )
        }
        #[test]
        fn shear_modulus() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model().shear_modulus(),
                get_thermoelastic_constitutive_model().shear_modulus()
            )
        }
        #[test]
        fn coefficient_of_thermal_expansion() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model()
                    .coefficient_of_thermal_expansion(),
                get_thermoelastic_constitutive_model().coefficient_of_thermal_expansion()
            )
        }
        #[test]
        fn reference_temperature() {
            assert_eq!(
                get_thermoelastic_thermal_conduction_constitutive_model().reference_temperature(),
                get_thermoelastic_constitutive_model().reference_temperature()
            )
        }
        #[test]
        fn cauchy_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .cauchy_stress(&get_deformation_gradient(), &get_temperature())?,
                &get_thermoelastic_constitutive_model()
                    .cauchy_stress(&get_deformation_gradient(), &get_temperature())?,
            )
        }
        #[test]
        fn cauchy_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .cauchy_tangent_stiffness(&get_deformation_gradient(), &get_temperature())?,
                &get_thermoelastic_constitutive_model()
                    .cauchy_tangent_stiffness(&get_deformation_gradient(), &get_temperature())?,
            )
        }
        #[test]
        fn first_piola_kirchhoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .first_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().first_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn first_piola_kirchhoff_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .first_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().first_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn heat_flux() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .heat_flux(&get_temperature_gradient()),
                &get_thermal_conduction_constitutive_model().heat_flux(&get_temperature_gradient()),
            )
        }
        #[test]
        fn second_piola_kirchhoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .second_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().second_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    &get_temperature(),
                )?,
            )
        }
        #[test]
        fn second_piola_kirchhoff_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &get_thermoelastic_thermal_conduction_constitutive_model()
                    .second_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        &get_temperature(),
                    )?,
                &get_thermoelastic_constitutive_model().second_piola_kirchhoff_stress(
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
                2 * std::mem::size_of::<&[Scalar; 1]>()
            )
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_constitutive_model;
