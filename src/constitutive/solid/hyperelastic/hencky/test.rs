use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Hencky {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
});
