use super::super::test::*;
use super::*;

test_solid_elastic_constitutive_model!(AlmansiHamel {
    bulk_modulus: ALMANSIHAMELPARAMETERS[0],
    shear_modulus: ALMANSIHAMELPARAMETERS[1],
});

crate::constitutive::solid::hyperelastic::test::test_solve!(AlmansiHamel {
    bulk_modulus: ALMANSIHAMELPARAMETERS[0],
    shear_modulus: ALMANSIHAMELPARAMETERS[1],
});
