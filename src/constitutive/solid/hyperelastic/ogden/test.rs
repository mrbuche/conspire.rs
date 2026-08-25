use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(Ogden {
    bulk_modulus: BULK_MODULUS,
    moduli: OGDEN_MODULI.to_vec(),
    exponents: OGDEN_EXPONENTS.to_vec(),
});
