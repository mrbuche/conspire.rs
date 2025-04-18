use super::super::test::*;
use super::*;

type AlmansiHamelType<'a> = AlmansiHamel<&'a [Scalar; 2]>;

test_solid_elastic_constitutive_model!(
    AlmansiHamelType,
    ALMANSIHAMELPARAMETERS,
    AlmansiHamel::new(ALMANSIHAMELPARAMETERS)
);

crate::constitutive::solid::hyperelastic::test::test_solve!(AlmansiHamel::new(
    ALMANSIHAMELPARAMETERS
));
