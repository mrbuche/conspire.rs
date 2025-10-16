use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Yeoh {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    extra_moduli: [-1.0, 3e-1, -1e-3, 1e-5,]
});
