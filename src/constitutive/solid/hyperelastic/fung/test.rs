use super::super::test::*;
use super::*;

type FungType<'a> = ArrudaBoyce<&'a [Scalar; 4]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(FungType, FUNGPARAMETERS, Fung::new(FUNGPARAMETERS));

test_minimize!(Fung::new(FUNGPARAMETERS));
test_solve!(Fung::new(FUNGPARAMETERS));

#[test]
fn extra_modulus() {
    assert_eq!(
        &FUNGPARAMETERS[2],
        Fung::new(FUNGPARAMETERS).extra_modulus()
    )
}

#[test]
fn exponent() {
    assert_eq!(&FUNGPARAMETERS[3], Fung::new(FUNGPARAMETERS).exponent())
}
