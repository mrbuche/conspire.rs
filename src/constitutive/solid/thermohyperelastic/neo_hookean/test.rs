use super::super::test::*;
use super::*;
use crate::math::assert::Assert;

use_thermoelastic_macros!();

test_solid_thermohyperelastic_constitutive_model!(NeoHookean {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
    reference_temperature: REFERENCE_TEMPERATURE,
});

mod consistency {
    use super::*;
    use crate::constitutive::solid::{
        elastic::Elastic,
        hyperelastic::{Hyperelastic, NeoHookean as HyperelasticNeoHookean},
    };
    fn model() -> NeoHookean {
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        }
    }
    fn hyperelastic_model() -> HyperelasticNeoHookean {
        HyperelasticNeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        }
    }
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            model().helmholtz_free_energy_density(
                &get_deformation_gradient(),
                REFERENCE_TEMPERATURE,
            )?,
            &hyperelastic_model().helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_stress() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &model().cauchy_stress(&get_deformation_gradient(), REFERENCE_TEMPERATURE)?,
            &hyperelastic_model().cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &model()
                .cauchy_tangent_stiffness(&get_deformation_gradient(), REFERENCE_TEMPERATURE)?,
            &hyperelastic_model().cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}

mod entropic {
    use super::*;
    use crate::{
        constitutive::solid::hyperelastic::{Hyperelastic, NeoHookean as HyperelasticNeoHookean},
        math::Rank2,
        units::{ReciprocalTemperature, Temperature},
    };
    fn model() -> NeoHookean {
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: ReciprocalTemperature::per_kelvin(0.0),
            reference_temperature: REFERENCE_TEMPERATURE,
        }
    }
    #[test]
    fn the_shear_stress_is_proportional_to_the_temperature() -> Result<(), AssertionError> {
        let cold = Temperature::kelvin(60.0);
        let hot = Temperature::kelvin(150.0);
        Assert::default().eq_within_tols(
            &(model()
                .cauchy_stress(&get_deformation_gradient(), hot)?
                .deviatoric()
                * 60.0),
            &(model()
                .cauchy_stress(&get_deformation_gradient(), cold)?
                .deviatoric()
                * 150.0),
        )
    }
    #[test]
    fn the_thermal_tangent_is_the_deviatoric_stress_over_the_temperature()
    -> Result<(), AssertionError> {
        let temperature = Temperature::kelvin(137.0);
        Assert::default().eq_within_tols(
            &(model().cauchy_thermal_tangent(&get_deformation_gradient(), temperature)?
                * temperature),
            &model()
                .cauchy_stress(&get_deformation_gradient(), temperature)?
                .deviatoric(),
        )
    }
    #[test]
    fn the_internal_energy_does_not_depend_on_the_shear_deformation() -> Result<(), AssertionError>
    {
        let temperature = Temperature::kelvin(137.0);
        let volumetric = HyperelasticNeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS * 0.0,
        }
        .helmholtz_free_energy_density(&get_deformation_gradient())?;
        let shear = HyperelasticNeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        }
        .helmholtz_free_energy_density(&get_deformation_gradient())?
            - volumetric;
        Assert::default().eq_within_tols(
            model().helmholtz_free_energy_density(&get_deformation_gradient(), temperature)?,
            &(volumetric + shear * (137.0 / REFERENCE_TEMPERATURE.value())),
        )
    }
}
