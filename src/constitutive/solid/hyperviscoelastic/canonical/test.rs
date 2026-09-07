use super::*;
use crate::{
    constitutive::{
        fluid::{
            hyperviscous::SaintVenantKirchhoff as ViscousSaintVenantKirchhoff, viscous::Viscous,
        },
        solid::{
            elastic_hyperviscous::ElasticHyperviscous, hyperelastic::SaintVenantKirchhoff,
            hyperviscoelastic::test::*, viscoelastic::Viscoelastic,
        },
    },
    math::{Rank2, assert::Assert},
    mechanics::{
        CauchyRateTangentStiffness, DeformationGradient, DeformationGradientRate,
        FirstPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffRateTangentStiffness,
    },
};

fn model() -> Canonical<SaintVenantKirchhoff, ViscousSaintVenantKirchhoff> {
    Canonical::from((
        SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        ViscousSaintVenantKirchhoff {
            bulk_viscosity: BULK_VISCOSITY,
            shear_viscosity: SHEAR_VISCOSITY,
        },
    ))
}

test_solid_hyperviscoelastic_constitutive_model!(model());

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::hyperelastic::Hyperelastic, mechanics::test::get_deformation_gradient,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &model().helmholtz_free_energy_density(&get_deformation_gradient())?,
            &SaintVenantKirchhoff {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            }
            .helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
}
