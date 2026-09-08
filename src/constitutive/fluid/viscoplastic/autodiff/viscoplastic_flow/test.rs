use super::super::Autodiff;
use super::AutodiffViscoplasticFlow;
use crate::{
    constitutive::fluid::viscoplastic::{Viscoplastic, ViscoplasticFlow},
    math::assert::{Assert, AssertionError},
    mechanics::MandelStressElastic,
    units::{Rate, Stress},
};

fn hand() -> ViscoplasticFlow {
    ViscoplasticFlow {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(1.0),
        rate_sensitivity: 0.25,
        reference_flow_rate: Rate::per_second(0.1),
    }
}

fn autodiff() -> Autodiff<AutodiffViscoplasticFlow> {
    Autodiff(AutodiffViscoplasticFlow {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(1.0),
        rate_sensitivity: 0.25,
        reference_flow_rate: Rate::per_second(0.1),
    })
}

fn deviatoric_mandel_stress() -> MandelStressElastic {
    MandelStressElastic::from([[1.3, 0.7, -0.4], [0.7, -0.9, 1.1], [-0.4, 1.1, -0.4]])
}

#[test]
fn matches_hand_written() -> Result<(), AssertionError> {
    let (ad, hand) = (autodiff(), hand());
    let yield_stress = hand.yield_stress;
    Assert::default().eq_within_tols(
        &ad.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?,
        &hand.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?,
    )?;
    Assert::default().eq_within_tols(
        &ad.dual_dissipation_potential(deviatoric_mandel_stress(), yield_stress)?,
        &hand.dual_dissipation_potential(deviatoric_mandel_stress(), yield_stress)?,
    )?;
    let plastic_stretching_rate =
        hand.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?;
    Assert::default().eq_within_tols(
        &ad.dissipation_potential(plastic_stretching_rate.clone(), yield_stress)?,
        &hand.dissipation_potential(plastic_stretching_rate, yield_stress)?,
    )
}

#[test]
fn canonical_matches_hand_written() -> Result<(), AssertionError> {
    use crate::{
        constitutive::{
            canonical::Canonical,
            solid::{
                elastic_viscoplastic::ElasticPlasticOrViscoplastic,
                hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
            },
        },
        mechanics::{DeformationGradientPlastic, test::get_deformation_gradient},
    };
    let bulk_modulus = Stress::pascals(1.3);
    let shear_modulus = Stress::pascals(0.7);
    let yield_stress = hand().yield_stress;
    let hand = Canonical::from((
        NeoHookean {
            bulk_modulus,
            shear_modulus,
        },
        hand(),
    ));
    let ad = Canonical::from((
        Autodiff(AutodiffNeoHookean {
            bulk_modulus,
            shear_modulus,
        }),
        autodiff(),
    ));
    let f = get_deformation_gradient();
    let f_p = DeformationGradientPlastic::from([
        [1.04, 0.02, -0.01],
        [0.0, 0.97, 0.03],
        [0.01, 0.0, 1.05],
    ]);
    Assert::default().eq_within_tols(
        &ad.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?,
        &hand.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?,
    )?;
    Assert::default().eq_within_tols(
        &ElasticPlasticOrViscoplastic::first_piola_kirchhoff_stress(&ad, &f, &f_p)?,
        &ElasticPlasticOrViscoplastic::first_piola_kirchhoff_stress(&hand, &f, &f_p)?,
    )
}
