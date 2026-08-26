use super::super::test::*;
use super::*;

test_solid_elastic_constitutive_model!(SethHillEulerian {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    exponent: EXPONENT,
});
