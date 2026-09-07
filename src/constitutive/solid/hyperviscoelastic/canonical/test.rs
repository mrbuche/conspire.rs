use super::*;
use crate::{
    constitutive::{
        fluid::hyperviscous::SaintVenantKirchhoff as ViscousSaintVenantKirchhoff,
        solid::hyperelastic::SaintVenantKirchhoff,
    },
    math::assert::{Assert, AssertionError},
    units::{Stress, Viscosity},
};

fn elastic() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn model() -> Canonical<SaintVenantKirchhoff, ViscousSaintVenantKirchhoff> {
    Canonical::from((
        elastic(),
        ViscousSaintVenantKirchhoff {
            bulk_viscosity: Viscosity::pascal_seconds(11.0),
            shear_viscosity: Viscosity::pascal_seconds(1.0),
        },
    ))
}

fn deformation_gradient() -> DeformationGradient {
    DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ])
}

#[test]
fn helmholtz_free_energy_density_matches_constituent() -> Result<(), AssertionError> {
    let f = deformation_gradient();
    Assert::default().eq_within_tols(
        &model().helmholtz_free_energy_density(&f)?,
        &elastic().helmholtz_free_energy_density(&f)?,
    )
}
