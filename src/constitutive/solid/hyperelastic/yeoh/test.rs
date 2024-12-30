use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(Yeoh, YEOHPARAMETERS, Yeoh::new(YEOHPARAMETERS));

test_solve!(Yeoh::new(YEOHPARAMETERS));

#[test]
fn get_moduli() {
    Yeoh::new(YEOHPARAMETERS)
        .get_moduli()
        .iter()
        .zip(YEOHPARAMETERS[1..].iter())
        .for_each(|(modulus_i, parameter_i)| assert_eq!(modulus_i, parameter_i))
}

#[test]
fn get_extra_moduli() {
    Yeoh::new(YEOHPARAMETERS)
        .get_extra_moduli()
        .iter()
        .zip(YEOHPARAMETERS[2..].iter())
        .for_each(|(modulus_i, parameter_i)| assert_eq!(modulus_i, parameter_i))
}
