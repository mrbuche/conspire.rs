use super::super::test::*;
use super::*;
use crate::math::assert::Assert;

test_solid_thermoelastic_constitutive_model!(AlmansiHamel {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
    reference_temperature: REFERENCE_TEMPERATURE,
});

mod consistency {
    use super::*;
    use crate::constitutive::solid::elastic::{AlmansiHamel as ElasticAlmansiHamel, Elastic};
    #[test]
    fn cauchy_stress() -> Result<(), AssertionError> {
        let model = AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        };
        let elastic_model = ElasticAlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        Assert::default().eq_within_tols(
            &model.cauchy_stress(&get_deformation_gradient(), model.reference_temperature())?,
            &elastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), AssertionError> {
        let model = AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        };
        let elastic_model = ElasticAlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        Assert::default().eq_within_tols(
            &model.cauchy_tangent_stiffness(
                &get_deformation_gradient(),
                model.reference_temperature(),
            )?,
            &elastic_model.cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}
