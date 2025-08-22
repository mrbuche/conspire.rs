use super::super::test::*;
use super::*;

type SaintVenantKirchhoffType<'a> = SaintVenantKirchhoff<&'a [Scalar; 2]>;

test_solid_elastic_constitutive_model!(
    SaintVenantKirchhoffType,
    SAINTVENANTKIRCHHOFFPARAMETERS,
    SaintVenantKirchhoff::new(SAINTVENANTKIRCHHOFFPARAMETERS)
);

crate::constitutive::solid::hyperelastic::test::test_solve!(SaintVenantKirchhoff::new(
    SAINTVENANTKIRCHHOFFPARAMETERS
));
