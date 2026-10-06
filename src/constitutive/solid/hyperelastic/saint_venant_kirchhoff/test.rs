use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(
    SaintVenantKirchhoff {
        bulk_modulus: BULK_MODULUS,
        shear_modulus: SHEAR_MODULUS,
    },
    (0.9, 0.95)
);

#[test]
fn biaxial_compression_is_not_a_minimum() {
    use crate::{
        constitutive::solid::{elastic::AppliedLoad, hyperelastic::SecondOrderMinimize},
        math::optimize::NewtonRaphson,
    };
    let model = SaintVenantKirchhoff {
        bulk_modulus: BULK_MODULUS,
        shear_modulus: SHEAR_MODULUS,
    };
    let result = model.minimize(
        AppliedLoad::BiaxialStress(0.77, 0.88),
        NewtonRaphson::default(),
    );
    assert!(result.is_err_and(|error| error.to_string().contains("not a minimum")))
}
