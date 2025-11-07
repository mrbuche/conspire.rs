use super::*;
use crate::constitutive::solid::thermohyperelastic::{
    SaintVenantKirchhoff,
    test::{BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE, SHEAR_MODULUS},
};

test_thermohyperelastic_thermal_conduction_constitutive_model!(
    ThermohyperelasticThermalConduction::from((
        SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        },
        Fourier {
            thermal_conductivity: THERMAL_CONDUCTIVITY
        }
    )),
    SaintVenantKirchhoff {
        bulk_modulus: BULK_MODULUS,
        shear_modulus: SHEAR_MODULUS,
        coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
        reference_temperature: REFERENCE_TEMPERATURE,
    },
    Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY
    }
);

macro_rules! test_thermohyperelastic_thermal_conduction_constitutive_model
{
    ($thermohyperelastic_thermal_conduction_constitutive_model: expr,
    $thermohyperelastic_constitutive_model: expr,
     $thermal_conduction_constitutive_model: expr) =>
    {
        use crate::{
            constitutive::multiphysics::solid_thermal::thermoelastic_thermal_conduction::test::test_thermoelastic_thermal_conduction_constitutive_model,
            math::test::{assert_eq, TestError}
        };
        test_thermoelastic_thermal_conduction_constitutive_model!(
            $thermohyperelastic_thermal_conduction_constitutive_model,
            $thermohyperelastic_constitutive_model,
            $thermal_conduction_constitutive_model
        );
        #[test]
        fn helmholtz_free_energy_density() -> Result<(), TestError>
        {
            assert_eq(
                &$thermohyperelastic_thermal_conduction_constitutive_model
                .helmholtz_free_energy_density(
                    &get_deformation_gradient(), get_temperature()
                )?,
                &$thermohyperelastic_constitutive_model
                .helmholtz_free_energy_density(
                    &get_deformation_gradient(), get_temperature()
                )?
            )
        }
    }
}
pub(crate) use test_thermohyperelastic_thermal_conduction_constitutive_model;
