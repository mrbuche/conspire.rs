use super::Torus;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::contract::{
            assert_against_samples, assert_contract, assert_lipschitz, grid, local, revolved,
        },
        solid::SolidOracle,
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};
use core::{array::from_fn, f64::consts::TAU};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn axis(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

fn upright() -> Torus {
    Torus::new(
        point([0.0; 3]),
        axis([0.0, 0.0, 1.0]),
        length(3.0),
        length(1.0),
    )
    .unwrap()
}

#[test]
fn signed_distance_is_negative_in_the_tube_and_measured_from_its_middle()
-> Result<(), AssertionError> {
    let torus = upright();
    let distance = |entries| torus.signed_distance(&point(entries));
    let assert = Assert::default();
    assert.eq_within_tols(distance([3.0, 0.0, 0.0]), &length(-1.0))?;
    assert.zero_within_tols(&distance([2.0, 0.0, 0.0]))?;
    assert.zero_within_tols(&distance([0.0, 4.0, 0.0]))?;
    assert.zero_within_tols(&distance([0.0, 3.0, 1.0]))?;
    assert.eq_within_tols(distance([0.0, 0.0, 0.0]), &length(2.0))?;
    assert.eq_within_tols(distance([10.0, 0.0, 0.0]), &length(6.0))
}

#[test]
fn projects_across_the_tube() -> Result<(), AssertionError> {
    let torus = upright();
    let assert = Assert::default();
    let found = torus.project(&point([5.0, 0.0, 0.0])).unwrap();
    assert.eq_within_tols(&found.point, &point([4.0, 0.0, 0.0]))?;
    assert.eq_within_tols(&found.normal, &Direction::from([1.0, 0.0, 0.0]))?;
    let above = torus.project(&point([3.0, 0.0, 5.0])).unwrap();
    assert.eq_within_tols(&above.point, &point([3.0, 0.0, 1.0]))
}

#[test]
fn on_the_axis_and_the_middle_circle_the_projection_is_still_on_the_surface()
-> Result<(), AssertionError> {
    let torus = upright();
    for entries in [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [3.0, 0.0, 0.0],
        [0.0, -3.0, 0.0],
    ] {
        let found = torus.project(&point(entries)).unwrap();
        Assert::default().zero_within_tols(&torus.signed_distance(&found.point))?;
    }
    Ok(())
}

#[test]
fn a_torus_that_is_not_a_ring_is_an_error() {
    let make = |major, minor, direction| {
        Torus::new(
            point([0.0; 3]),
            axis(direction),
            length(major),
            length(minor),
        )
    };
    let up = [0.0, 0.0, 1.0];
    assert!(make(3.0, 1.0, up).is_ok());
    assert!(make(1.0, 1.0, up).is_err());
    assert!(make(1.0, 2.0, up).is_err());
    assert!(make(3.0, 0.0, up).is_err());
    assert!(make(3.0, -1.0, up).is_err());
    assert!(make(3.0, 1.0, [0.0; 3]).is_err());
}

type Case = ([f64; 3], [f64; 3], f64, f64);

const CASES: [Case; 5] = [
    ([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0),
    ([1.0, -1.0, 0.5], [1.0, 1.0, 0.0], 2.0, 0.5),
    ([0.0; 3], [1.0, 2.0, 3.0], 1.0, 0.9),
    ([0.0; 3], [0.0, 1.0, 0.0], 3e-9, 1e-9),
    ([4e8, 0.0, -1e8], [1.0, 0.0, 0.0], 3e8, 1e8),
];

fn make(case: Case) -> Torus {
    let (center, direction, major, minor) = case;
    Torus::new(point(center), axis(direction), length(major), length(minor)).unwrap()
}

fn around(case: Case) -> ([f64; 3], [f64; 3]) {
    let (center, _, major, minor) = case;
    let reach = major + minor;
    (center.map(|x| x - reach), center.map(|x| x + reach))
}

#[test]
fn satisfies_the_contract_for_any_size_place_and_direction() -> Result<(), AssertionError> {
    for case in CASES {
        let torus = make(case);
        let (center, direction, major, minor) = case;
        let unit = axis(direction).normalized();
        let (low, high) = around(case);
        let points = grid(low, high, 9);
        let inside = |p: &Coordinate<3>| {
            let (along, across) = local(&point(center), &unit, p);
            let tube = Coordinate::from([across - length(major), along]).norm();
            tube < length(minor)
        };
        assert_contract(&torus, &points, minor, inside)?;
        assert_lipschitz(&torus, &points, minor)?;
    }
    Ok(())
}

#[test]
fn agrees_with_a_sampling_of_its_surface() -> Result<(), AssertionError> {
    for case in &CASES[..3] {
        let torus = make(*case);
        let (center, direction, major, minor) = *case;
        let profile: Vec<[Quantity<Length>; 2]> = (0..240)
            .map(|i| {
                let angle = TAU * i as f64 / 240.0;
                [
                    length(major + minor * angle.cos()),
                    length(minor * angle.sin()),
                ]
            })
            .collect();
        let angles = 240;
        let samples = revolved(
            &point(center),
            &axis(direction).normalized(),
            &profile,
            angles,
        );
        let step = TAU * minor / 240.0;
        let arc = TAU * (major + minor) / angles as f64;
        let spacing = 0.55 * step.hypot(arc);
        let (low, high) = around(*case);
        assert_against_samples(&torus, &grid(low, high, 5), &samples, spacing)?;
    }
    Ok(())
}

#[test]
fn the_middle_circle_is_what_it_is_measured_from() -> Result<(), AssertionError> {
    let torus = upright();
    for i in 0..16 {
        let angle = TAU * i as f64 / 16.0;
        let on_circle = point(from_fn(|k| [3.0 * angle.cos(), 3.0 * angle.sin(), 0.0][k]));
        Assert::default().eq_within_tols(torus.signed_distance(&on_circle), &length(-1.0))?;
    }
    Ok(())
}
