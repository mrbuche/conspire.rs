use super::super::test::*;
use super::*;

type MooneyRivlinType<'a> = MooneyRivlin<&'a [Scalar; 3]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    MooneyRivlinType,
    MOONEYRIVLINPARAMETERS,
    MooneyRivlin::new(MOONEYRIVLINPARAMETERS)
);

test_solve!(MooneyRivlin::new(MOONEYRIVLINPARAMETERS));

#[test]
fn extra_modulus() {
    assert_eq!(
        &MOONEYRIVLINPARAMETERS[2],
        MooneyRivlin::new(MOONEYRIVLINPARAMETERS).extra_modulus()
    )
}
