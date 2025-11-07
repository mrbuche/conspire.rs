use super::*;
use crate::constitutive::solid::thermoelastic::{
    AlmansiHamel,
    test::{BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE, SHEAR_MODULUS},
};

test_thermoelastic_thermal_conduction_constitutive_model!(
    ThermoelasticThermalConduction::construct(
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        },
        Fourier {
            thermal_conductivity: THERMAL_CONDUCTIVITY
        },
    ),
    AlmansiHamel {
        bulk_modulus: BULK_MODULUS,
        shear_modulus: SHEAR_MODULUS,
        coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
        reference_temperature: REFERENCE_TEMPERATURE,
    },
    Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY
    }
);

macro_rules! test_thermoelastic_thermal_conduction_constitutive_model {
    ($thermoelastic_thermal_conduction_constitutive_model: expr,
     $thermoelastic_constitutive_model: expr,
     $thermal_conduction_constitutive_model: expr) => {
        use crate::{
            constitutive::thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
            mechanics::test::{
                get_deformation_gradient, get_temperature, get_temperature_gradient,
            },
        };
        #[test]
        fn bulk_modulus() {
            assert_eq!(
                $thermoelastic_thermal_conduction_constitutive_model.bulk_modulus(),
                $thermoelastic_constitutive_model.bulk_modulus()
            )
        }
        #[test]
        fn shear_modulus() {
            assert_eq!(
                $thermoelastic_thermal_conduction_constitutive_model.shear_modulus(),
                $thermoelastic_constitutive_model.shear_modulus()
            )
        }
        #[test]
        fn coefficient_of_thermal_expansion() {
            assert_eq!(
                $thermoelastic_thermal_conduction_constitutive_model
                    .coefficient_of_thermal_expansion(),
                $thermoelastic_constitutive_model.coefficient_of_thermal_expansion()
            )
        }
        #[test]
        fn reference_temperature() {
            assert_eq!(
                $thermoelastic_thermal_conduction_constitutive_model.reference_temperature(),
                $thermoelastic_constitutive_model.reference_temperature()
            )
        }
        #[test]
        fn cauchy_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .cauchy_stress(&get_deformation_gradient(), get_temperature())?,
                &$thermoelastic_constitutive_model
                    .cauchy_stress(&get_deformation_gradient(), get_temperature())?,
            )
        }
        #[test]
        fn cauchy_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .cauchy_tangent_stiffness(&get_deformation_gradient(), get_temperature())?,
                &$thermoelastic_constitutive_model
                    .cauchy_tangent_stiffness(&get_deformation_gradient(), get_temperature())?,
            )
        }
        #[test]
        fn first_piola_kirchhoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .first_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        get_temperature(),
                    )?,
                &$thermoelastic_constitutive_model.first_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    get_temperature(),
                )?,
            )
        }
        #[test]
        fn first_piola_kirchhoff_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .first_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        get_temperature(),
                    )?,
                &$thermoelastic_constitutive_model.first_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    get_temperature(),
                )?,
            )
        }
        #[test]
        fn heat_flux() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .heat_flux(&get_temperature_gradient()),
                &$thermal_conduction_constitutive_model.heat_flux(&get_temperature_gradient()),
            )
        }
        #[test]
        fn second_piola_kirchhoff_stress() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .second_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        get_temperature(),
                    )?,
                &$thermoelastic_constitutive_model.second_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    get_temperature(),
                )?,
            )
        }
        #[test]
        fn second_piola_kirchhoff_tangent_stiffness() -> Result<(), crate::math::test::TestError> {
            crate::math::test::assert_eq(
                &$thermoelastic_thermal_conduction_constitutive_model
                    .second_piola_kirchhoff_stress(
                        &get_deformation_gradient(),
                        get_temperature(),
                    )?,
                &$thermoelastic_constitutive_model.second_piola_kirchhoff_stress(
                    &get_deformation_gradient(),
                    get_temperature(),
                )?,
            )
        }
        // #[test]
        // fn size() {
        //     assert_eq!(
        //         std::mem::size_of::<
        //             ThermoelasticThermalConduction<
        //                 $thermoelastic_constitutive_model,
        //                 $thermal_conduction_constitutive_model,
        //             >,
        //         >(),
        //         2 * std::mem::size_of::<&[Scalar; 1]>()
        //     )
        // }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_constitutive_model;
