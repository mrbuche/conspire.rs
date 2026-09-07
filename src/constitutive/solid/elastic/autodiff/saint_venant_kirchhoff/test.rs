use super::AutodiffSaintVenantKirchhoff;
use crate::{
    constitutive::solid::elastic::{
        Elastic, SaintVenantKirchhoff,
        autodiff::{
            Autodiff,
            test::{assert_close_2, assert_close_4, ok},
        },
    },
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
fn matches_hand_written() {
    let (ad, hand, f) = (autodiff(), hand(), get_deformation_gradient());
    assert_close_2(&ok(ad.cauchy_stress(&f)), &ok(hand.cauchy_stress(&f)), 1e-8);
    assert_close_2(
        &ok(ad.first_piola_kirchhoff_stress(&f)),
        &ok(hand.first_piola_kirchhoff_stress(&f)),
        1e-8,
    );
    assert_close_2(
        &ok(ad.second_piola_kirchhoff_stress(&f)),
        &ok(hand.second_piola_kirchhoff_stress(&f)),
        1e-8,
    );
    assert_close_4(
        &ok(ad.cauchy_tangent_stiffness(&f)),
        &ok(hand.cauchy_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &ok(ad.first_piola_kirchhoff_tangent_stiffness(&f)),
        &ok(hand.first_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &ok(ad.second_piola_kirchhoff_tangent_stiffness(&f)),
        &ok(hand.second_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
}
