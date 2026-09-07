use super::*;
use crate::{
    constitutive::{
        fluid::hyperviscous::Newtonian,
        solid::{
            elastic::{AlmansiHamelEulerian, Elastic},
            viscoelastic::Viscoelastic,
        },
    },
    math::{
        TensorArray,
        assert::{Assert, AssertionError},
    },
    units::{Stress, Viscosity},
};

fn flow() -> Newtonian {
    Newtonian {
        bulk_viscosity: Viscosity::pascal_seconds(11.0),
        shear_viscosity: Viscosity::pascal_seconds(1.0),
    }
}

fn elastic() -> AlmansiHamelEulerian {
    AlmansiHamelEulerian {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn model() -> Canonical<AlmansiHamelEulerian, Newtonian> {
    Canonical::from((elastic(), flow()))
}

fn deformation_gradient() -> DeformationGradient {
    DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ])
}

fn deformation_gradient_rate() -> DeformationGradientRate {
    DeformationGradientRate::from([
        [0.16276008, 0.16544806, 0.10516932],
        [0.11349288, 0.16559786, 0.13899089],
        [0.19497108, 0.11119965, 0.19226318],
    ])
}

#[test]
fn dissipation_non_negative() -> Result<(), AssertionError> {
    let (model, f, f_rate) = (model(), deformation_gradient(), deformation_gradient_rate());
    Assert::non_negative(&model.viscous_dissipation(&f, &f_rate)?)?;
    Assert::non_negative(&model.internal_dissipation(&f, &f_rate)?)
}

#[test]
fn viscous_dissipation_matches_constituent() -> Result<(), AssertionError> {
    let (model, f, f_rate) = (model(), deformation_gradient(), deformation_gradient_rate());
    Assert::default().eq_within_tols(
        &model.viscous_dissipation(&f, &f_rate)?,
        &flow().viscous_dissipation(&f, &f_rate)?,
    )
}

#[test]
fn zero_rate_reduces_to_elastic() -> Result<(), AssertionError> {
    let f = deformation_gradient();
    Assert::default().eq_within_tols(
        &model().cauchy_stress(&f, &DeformationGradientRate::zero())?,
        &elastic().cauchy_stress(&f)?,
    )
}
