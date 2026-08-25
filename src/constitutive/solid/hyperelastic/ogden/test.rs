use super::super::test::*;
use super::*;

const NUM_OGDEN_TERMS: usize = 3;
const OGDEN_MODULI: [Quantity<Stress>; NUM_OGDEN_TERMS] = [
    Stress::pascals(2.3),
    Stress::pascals(1.1e-2),
    Stress::pascals(-2.0e-3),
];
const OGDEN_EXPONENTS: [Scalar; NUM_OGDEN_TERMS] = [1.3, 5.0, -2.0];

test_solid_hyperelastic_constitutive_model!(Ogden {
    bulk_modulus: BULK_MODULUS,
    moduli: OGDEN_MODULI,
    exponents: OGDEN_EXPONENTS,
});
