use super::super::test::*;
use super::*;

test_solid_elastic_constitutive_model!(
    AlmansiHamel,
    ALMANSIHAMELPARAMETERS,
    AlmansiHamel::new(ALMANSIHAMELPARAMETERS)
);

crate::constitutive::solid::hyperelastic::test::test_solve!(AlmansiHamel::new(
    ALMANSIHAMELPARAMETERS
));
