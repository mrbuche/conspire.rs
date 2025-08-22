use super::super::test::*;
use super::*;
use crate::mechanics::CauchyTangentStiffness;

type SaintVenantKirchhoffType<'a> = SaintVenantKirchhoff<&'a [Scalar; 2]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    SaintVenantKirchhoffType,
    SAINTVENANTKIRCHHOFFPARAMETERS,
    SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS)
);

test_minimize!(SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS));
test_solve!(SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS));
