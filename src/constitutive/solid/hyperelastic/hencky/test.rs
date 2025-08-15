use super::super::test::*;
use super::*;
use crate::mechanics::CauchyTangentStiffness;

type HenckyType<'a> = Hencky<&'a [Scalar; 2]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    HenckyType,
    HENCKYPARAMETERS,
    Hencky::new(HENCKYPARAMETERS)
);

test_minimize!(Hencky::new(HENCKYPARAMETERS));
test_solve!(Hencky::new(HENCKYPARAMETERS));
