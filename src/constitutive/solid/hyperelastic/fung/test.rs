use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Fung {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    extra_modulus: EXTRA_MODULUS,
    exponent: EXPONENT,
});
