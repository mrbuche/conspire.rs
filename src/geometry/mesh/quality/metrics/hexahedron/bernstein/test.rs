use super::{certifies, coefficients, margin, sampled_minimum};
use crate::{
    geometry::Coordinates,
    math::{Quantity, Scalar},
};

const ELEMENT: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

fn cube(side: Scalar) -> Coordinates<3> {
    Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side, side, 0.0],
        [0.0, side, 0.0],
        [0.0, 0.0, side],
        [side, 0.0, side],
        [side, side, side],
        [0.0, side, side],
    ])
}

fn perturbed(seed: &mut u64, amplitude: Scalar) -> Coordinates<3> {
    let mut random = || {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as Scalar / (1u64 << 31) as Scalar - 1.0) * amplitude
    };
    let mut coordinates = cube(1.0);
    (0..8).for_each(|node| (0..3).for_each(|d| coordinates[node][d] += Quantity::new(random())));
    coordinates
}

#[test]
fn a_cube_is_certified_with_every_coefficient_its_volume() {
    coefficients(&ELEMENT, &cube(1.0))
        .iter()
        .for_each(|&coefficient| assert!((coefficient - 1.0).abs() < 1e-12, "{coefficient}"));
    coefficients(&ELEMENT, &cube(0.5))
        .iter()
        .for_each(|&coefficient| assert!((coefficient - 0.125).abs() < 1e-12, "{coefficient}"));
    assert!(certifies(&ELEMENT, &cube(1.0)));
    assert!((margin(&ELEMENT, &cube(0.5)) - 1.0).abs() < 1e-12);
}

#[test]
fn an_inverted_element_is_not_certified() {
    let mut coordinates = cube(1.0);
    coordinates[6][2] = Quantity::new(-3.0);
    assert!(!certifies(&ELEMENT, &coordinates));
    assert!(margin(&ELEMENT, &coordinates) < 0.0);
}

/// Certification must imply the determinant is positive throughout, not
/// merely where it happens to be sampled.
#[test]
fn certification_is_sound() {
    let mut seed = 0x5eed;
    let mut certified = 0;
    (0..300).for_each(|_| {
        let coordinates = perturbed(&mut seed, 0.7);
        if certifies(&ELEMENT, &coordinates) {
            certified += 1;
            let sampled = sampled_minimum(&ELEMENT, &coordinates, 12);
            assert!(sampled > 0.0, "certified but sampled {sampled}");
        }
    });
    assert!(certified > 20, "only {certified} certified");
}

/// The corner Jacobians the existing metrics use miss inversions away from
/// the corners, which is the whole reason to certify instead of sample.
#[test]
fn corners_alone_can_miss_what_certification_catches() {
    let mut seed = 0xc0ffee;
    let mut missed = 0;
    (0..600).for_each(|_| {
        let coordinates = perturbed(&mut seed, 1.1);
        let corners = super::super::minimum_scaled_jacobian(&ELEMENT, &coordinates);
        if corners > 0.0 && sampled_minimum(&ELEMENT, &coordinates, 12) <= 0.0 {
            missed += 1;
            assert!(!certifies(&ELEMENT, &coordinates));
        }
    });
    assert!(missed > 0, "the fixture never produced such an element");
}
