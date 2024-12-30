use super::super::test::*;
use super::*;

use_thermoelastic_macros!();

test_solid_thermohyperelastic_constitutive_model!(
    SaintVenantKirchoff,
    SAINTVENANTKIRCHOFFPARAMETERS,
    SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS)
);

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::{
            elastic::Elastic,
            hyperelastic::{
                test::SAINTVENANTKIRCHOFFPARAMETERS as HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS,
                Hyperelastic, SaintVenantKirchoff as HyperelasticSaintVenantKirchoff,
            },
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), TestError> {
        let model = SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchoff::new(HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_helmholtz_free_energy_density(
                &get_deformation_gradient(),
                model.get_reference_temperature(),
            )?,
            &hyperelastic_model
                .calculate_helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchoff::new(HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_cauchy_stress(
                &get_deformation_gradient(),
                model.get_reference_temperature(),
            )?,
            &hyperelastic_model.calculate_cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), TestError> {
        let model = SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchoff::new(HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_cauchy_tangent_stiffness(
                &get_deformation_gradient(),
                model.get_reference_temperature(),
            )?,
            &hyperelastic_model.calculate_cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}
