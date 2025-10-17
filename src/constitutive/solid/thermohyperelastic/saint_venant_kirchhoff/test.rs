use super::super::test::*;
use super::*;

use_thermoelastic_macros!();

test_solid_thermohyperelastic_constitutive_model!(SaintVenantKirchhoff {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
    reference_temperature: REFERENCE_TEMPERATURE,
});

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::{
            elastic::Elastic,
            hyperelastic::{
                Hyperelastic, SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
            },
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        };
        let hyperelastic_model = HyperelasticSaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.helmholtz_free_energy_density(
                &get_deformation_gradient(),
                model.reference_temperature(),
            )?,
            &hyperelastic_model.helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        };
        let hyperelastic_model = HyperelasticSaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.cauchy_stress(&get_deformation_gradient(), model.reference_temperature())?,
            &hyperelastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
            reference_temperature: REFERENCE_TEMPERATURE,
        };
        let hyperelastic_model = HyperelasticSaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.cauchy_tangent_stiffness(
                &get_deformation_gradient(),
                model.reference_temperature(),
            )?,
            &hyperelastic_model.cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}
