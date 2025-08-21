use super::super::test::*;
use super::*;
use crate::mechanics::CauchyTangentStiffness;

type HenckyType<'a> = Hencky<&'a [Scalar; 2]>;

test_solid_elastic_constitutive_model!(HenckyType, HENCKYPARAMETERS, Hencky::new(HENCKYPARAMETERS));

crate::constitutive::solid::hyperelastic::test::test_solve!(Hencky::new(HENCKYPARAMETERS));
