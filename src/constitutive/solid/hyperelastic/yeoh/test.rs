use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Yeoh {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    extra_moduli: YEOH_EXTRA_MODULI,
});
