use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Yeoh {
    bulk_modulus: BULK_MODULUS,
    moduli: YEOH_MODULI.to_vec(),
});
