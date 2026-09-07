use super::AutodiffSaintVenantKirchhoff;
use crate::{
    constitutive::solid::elastic::{Elastic, SaintVenantKirchhoff, autodiff::Autodiff},
    math::assert::{Assert, AssertionError},
    mechanics::test::get_deformation_gradient,
    units::Stress,
};

fn hand() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    }
}

fn autodiff() -> Autodiff<AutodiffSaintVenantKirchhoff> {
    Autodiff(AutodiffSaintVenantKirchhoff {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    })
}

#[test]
fn matches_hand_written() -> Result<(), AssertionError> {
    let (ad, hand, f) = (autodiff(), hand(), get_deformation_gradient());
    Assert::default().eq_within_tols(&ad.cauchy_stress(&f)?, &hand.cauchy_stress(&f)?)?;
    Assert::default().eq_within_tols(
        &ad.first_piola_kirchhoff_stress(&f)?,
        &hand.first_piola_kirchhoff_stress(&f)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.second_piola_kirchhoff_stress(&f)?,
        &hand.second_piola_kirchhoff_stress(&f)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.cauchy_tangent_stiffness(&f)?,
        &hand.cauchy_tangent_stiffness(&f)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.first_piola_kirchhoff_tangent_stiffness(&f)?,
        &hand.first_piola_kirchhoff_tangent_stiffness(&f)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.second_piola_kirchhoff_tangent_stiffness(&f)?,
        &hand.second_piola_kirchhoff_tangent_stiffness(&f)?,
    )
}
