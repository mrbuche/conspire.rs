use super::super::test::*;
use super::*;

use_elastic_hyperviscous_macros!();

test_solid_hyperviscoelastic_constitutive_model!(
    SaintVenantKirchhoff,
    SAINTVENANTKIRCHOFFPARAMETERS,
    SaintVenantKirchhoff::new(SAINTVENANTKIRCHOFFPARAMETERS)
);

test_solve!(SaintVenantKirchhoff::new(SAINTVENANTKIRCHOFFPARAMETERS));

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::{
            elastic::Elastic,
            hyperelastic::{
                test::SAINTVENANTKIRCHOFFPARAMETERS as HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS,
                Hyperelastic, SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
            },
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff::new(SAINTVENANTKIRCHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchhoff::new(HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.helmholtz_free_energy_density(&get_deformation_gradient())?,
            &hyperelastic_model.helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff::new(SAINTVENANTKIRCHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchhoff::new(HYPERELASTICSAINTVENANTKIRCHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.cauchy_stress(
                &get_deformation_gradient(),
                &DeformationGradientRate::zero(),
            )?,
            &hyperelastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
}
