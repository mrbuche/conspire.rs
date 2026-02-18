use crate::{
    constitutive::{
        hybrid::ElasticMultiplicative,
        solid::{
            elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            hyperelastic::{NeoHookean, SaintVenantKirchhoff},
        },
    },
    math::TestError,
};

use crate::{
    constitutive::solid::elastic::AppliedLoad,
    math::optimize::{GradientDescent, NewtonRaphson},
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
        SaintVenantKirchhoff {
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
        SaintVenantKirchhoff {
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
