use super::super::test::*;
use super::*;

pub const MIXING_PARAMETER: Scalar = 0.35;

test_solid_hyperelastic_constitutive_model!(BlatzKo {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    mixing_parameter: MIXING_PARAMETER,
});
