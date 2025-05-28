use super::super::test::*;
use super::*;

type NeoHookeanType<'a> = NeoHookean<&'a [Scalar; 2]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    NeoHookeanType,
    NEOHOOKEANPARAMETERS,
    NeoHookean::new(NEOHOOKEANPARAMETERS)
);

test_minimize!(NeoHookean::new(NEOHOOKEANPARAMETERS));
test_solve!(NeoHookean::new(NEOHOOKEANPARAMETERS));
