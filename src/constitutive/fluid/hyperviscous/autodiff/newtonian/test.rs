use super::super::Autodiff;
use super::AutodiffNewtonian;
use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::{
            hyperviscous::{Hyperviscous, Newtonian},
            viscous::Viscous,
        },
        solid::{
            hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
            viscoelastic::Viscoelastic,
        },
    },
    math::assert::{Assert, AssertionError},
    mechanics::test::{get_deformation_gradient, get_deformation_gradient_rate},
    units::{Stress, Viscosity},
};

fn hand() -> Newtonian {
    Newtonian {
        bulk_viscosity: Viscosity::pascal_seconds(1.3),
        shear_viscosity: Viscosity::pascal_seconds(0.7),
    }
}

fn autodiff() -> Autodiff<AutodiffNewtonian> {
    Autodiff(AutodiffNewtonian {
        bulk_viscosity: Viscosity::pascal_seconds(1.3),
        shear_viscosity: Viscosity::pascal_seconds(0.7),
    })
}

#[test]
fn matches_hand_written() -> Result<(), AssertionError> {
    let (ad, hand) = (autodiff(), hand());
    let (f, f_dot) = (get_deformation_gradient(), get_deformation_gradient_rate());
    Assert::default().eq_within_tols(
        &ad.viscous_cauchy_stress(&f, &f_dot)?,
        &hand.viscous_cauchy_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_first_piola_kirchhoff_stress(&f, &f_dot)?,
        &hand.viscous_first_piola_kirchhoff_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_second_piola_kirchhoff_stress(&f, &f_dot)?,
        &hand.viscous_second_piola_kirchhoff_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_cauchy_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.viscous_cauchy_rate_tangent_stiffness(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_first_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.viscous_first_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_second_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.viscous_second_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.viscous_dissipation(&f, &f_dot)?,
        &hand.viscous_dissipation(&f, &f_dot)?,
    )
}

#[test]
fn canonical_viscoelastic_matches_hand_written() -> Result<(), AssertionError> {
    let (bulk_modulus, shear_modulus) = (Stress::pascals(1.3), Stress::pascals(0.7));
    let (bulk_viscosity, shear_viscosity) = (
        Viscosity::pascal_seconds(1.1),
        Viscosity::pascal_seconds(0.5),
    );
    let hand = Canonical::from((
        NeoHookean {
            bulk_modulus,
            shear_modulus,
        },
        Newtonian {
            bulk_viscosity,
            shear_viscosity,
        },
    ));
    let ad = Canonical::from((
        Autodiff(AutodiffNeoHookean {
            bulk_modulus,
            shear_modulus,
        }),
        Autodiff(AutodiffNewtonian {
            bulk_viscosity,
            shear_viscosity,
        }),
    ));
    let (f, f_dot) = (get_deformation_gradient(), get_deformation_gradient_rate());
    Assert::default().eq_within_tols(
        &ad.cauchy_stress(&f, &f_dot)?,
        &hand.cauchy_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.first_piola_kirchhoff_stress(&f, &f_dot)?,
        &hand.first_piola_kirchhoff_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.second_piola_kirchhoff_stress(&f, &f_dot)?,
        &hand.second_piola_kirchhoff_stress(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.cauchy_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.cauchy_rate_tangent_stiffness(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.first_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.first_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.second_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
        &hand.second_piola_kirchhoff_rate_tangent_stiffness(&f, &f_dot)?,
    )
}
