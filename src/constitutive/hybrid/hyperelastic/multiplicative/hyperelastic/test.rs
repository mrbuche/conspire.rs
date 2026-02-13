use crate::{
    constitutive::{
        hybrid::ElasticMultiplicative,
        solid::{
            elastic::{
                AlmansiHamel,
                test::{BULK_MODULUS, SHEAR_MODULUS},
            },
            hyperelastic::NeoHookean,
        },
    },
    math::{TensorArray, TestError},
};

use crate::{
    constitutive::solid::elastic::{AppliedLoad, internal_variables::ElasticIV},
    math::{
        TensorRank4,
        optimize::{GradientDescent, NewtonRaphson},
        test::{ErrorTensor, assert_eq_from_fd},
    },
    mechanics::*,
};

const STRETCH: Scalar = 1.5;

#[test]
fn minimize_1() -> Result<(), TestError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::FirstOrderMinimize;
    let model = ElasticMultiplicative::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_f, _f_2) = model.minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn minimize_2() -> Result<(), TestError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::SecondOrderMinimize;
    let model = ElasticMultiplicative::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_f, _f_2) = model.minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        NewtonRaphson::default(),
    )?;
    Ok(())
}
