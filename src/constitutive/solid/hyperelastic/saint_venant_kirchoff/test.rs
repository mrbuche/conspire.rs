use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    SaintVenantKirchoff,
    SAINTVENANTKIRCHOFFPARAMETERS,
    SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS)
);

test_solve!(SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS));
