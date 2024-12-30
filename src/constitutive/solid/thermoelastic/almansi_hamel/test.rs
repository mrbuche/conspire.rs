use super::super::test::*;
use super::*;

test_solid_thermoelastic_constitutive_model!(
    AlmansiHamel,
    ALMANSIHAMELPARAMETERS,
    AlmansiHamel::new(ALMANSIHAMELPARAMETERS)
);

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::elastic::{
            test::ALMANSIHAMELPARAMETERS as ELASTICALMANSIHAMELPARAMETERS,
            AlmansiHamel as ElasticAlmansiHamel, Elastic,
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = AlmansiHamel::new(ALMANSIHAMELPARAMETERS);
        let elastic_model = ElasticAlmansiHamel::new(ELASTICALMANSIHAMELPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_cauchy_stress(
                &get_deformation_gradient(),
                model.get_reference_temperature(),
            )?,
            &elastic_model.calculate_cauchy_stress(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() -> Result<(), TestError> {
        let model = AlmansiHamel::new(ALMANSIHAMELPARAMETERS);
        let elastic_model = ElasticAlmansiHamel::new(ELASTICALMANSIHAMELPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_cauchy_tangent_stiffness(
                &get_deformation_gradient(),
                model.get_reference_temperature(),
            )?,
            &elastic_model.calculate_cauchy_tangent_stiffness(&get_deformation_gradient())?,
        )
    }
}
