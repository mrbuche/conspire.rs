use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(Fung, FUNGPARAMETERS, Fung::new(FUNGPARAMETERS));

test_solve!(Fung::new(FUNGPARAMETERS));

#[test]
fn get_extra_modulus() {
    assert_eq!(
        &FUNGPARAMETERS[2],
        Fung::new(FUNGPARAMETERS).get_extra_modulus()
    )
}

#[test]
fn get_exponent() {
    assert_eq!(&FUNGPARAMETERS[3], Fung::new(FUNGPARAMETERS).get_exponent())
}
