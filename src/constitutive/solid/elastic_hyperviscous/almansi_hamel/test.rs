use super::super::test::*;
use super::*;

use crate::{
    math::Tensor,
    mechanics::{
        CauchyTangentStiffness, FirstPiolaKirchhoffRateTangentStiffness,
        FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffRateTangentStiffness,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

test_solid_elastic_hyperviscous_constitutive_model!(AlmansiHamel {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    bulk_viscosity: BULK_VISCOSITY,
    shear_viscosity: SHEAR_VISCOSITY,
});

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::elastic::{AlmansiHamel as ElasticAlmansiHamel, Elastic},
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            bulk_viscosity: BULK_VISCOSITY,
            shear_viscosity: SHEAR_VISCOSITY,
        };
        let hyperelastic_model = ElasticAlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.cauchy_stress(
                &get_deformation_gradient(),
                &DeformationGradientRate::zero(),
            )?,
            &hyperelastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
}
