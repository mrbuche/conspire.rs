use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    MooneyRivlin,
    MOONEYRIVLINPARAMETERS,
    MooneyRivlin::new(MOONEYRIVLINPARAMETERS)
);

test_solve!(MooneyRivlin::new(MOONEYRIVLINPARAMETERS));

#[test]
fn get_extra_modulus() {
    assert_eq!(
        &MOONEYRIVLINPARAMETERS[2],
        MooneyRivlin::new(MOONEYRIVLINPARAMETERS).get_extra_modulus()
    )
}
