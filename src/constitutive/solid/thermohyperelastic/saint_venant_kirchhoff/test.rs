use super::super::test::*;
use super::*;

type SaintVenantKirchhoffType<'a> = SaintVenantKirchhoff<&'a [Scalar; 4]>;

use_thermoelastic_macros!();

test_solid_thermohyperelastic_constitutive_model!(
    SaintVenantKirchhoffType,
    SAINTVENANTKIRCHHOFFPARAMETERS,
    SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS)
);

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::{
            elastic::Elastic,
            hyperelastic::{
                Hyperelastic, SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
                test::SAINTVENANTKIRCHHOFFPARAMETERS as HYPERELASTICSAINTVENANTKIRCHHOFFPARAMETERS,
            },
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchhoff::new(HYPERELASTICSAINTVENANTKIRCHHOFFPARAMETERS);
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
        let model = SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchhoff::new(HYPERELASTICSAINTVENANTKIRCHHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.cauchy_stress(&get_deformation_gradient(), model.reference_temperature())?,
            &hyperelastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS);
        let hyperelastic_model =
            HyperelasticSaintVenantKirchhoff::new(HYPERELASTICSAINTVENANTKIRCHHOFFPARAMETERS);
        assert_eq_within_tols(
            &model.cauchy_tangent_stiffness(
                &get_deformation_gradient(),
                model.reference_temperature(),
            )?,
            &hyperelastic_model.cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}
