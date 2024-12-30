use super::super::test::*;
use super::*;

use_viscoelastic_macros!();

test_solid_elastic_hyperviscous_constitutive_model!(
    AlmansiHamel,
    ALMANSIHAMELPARAMETERS,
    AlmansiHamel::new(ALMANSIHAMELPARAMETERS)
);

test_solve!(AlmansiHamel::new(ALMANSIHAMELPARAMETERS));

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
        let hyperelastic_model = ElasticAlmansiHamel::new(ELASTICALMANSIHAMELPARAMETERS);
        assert_eq_within_tols(
            &model.calculate_cauchy_stress(
                &get_deformation_gradient(),
                &DeformationGradientRate::zero(),
            )?,
            &hyperelastic_model.calculate_cauchy_stress(&get_deformation_gradient())?,
        )
    }
}
