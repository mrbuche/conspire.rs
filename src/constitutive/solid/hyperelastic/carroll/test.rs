use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Carroll {
    bulk_modulus: BULK_MODULUS,
    linear_modulus: LINEAR_MODULUS,
    quartic_modulus: QUARTIC_MODULUS,
    second_invariant_modulus: SECOND_INVARIANT_MODULUS,
});
