use super::Cone;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::contract::{
            along_polyline, assert_against_samples, assert_contract, assert_lipschitz, grid, local,
            revolved,
        },
        solid::SolidOracle,
    },
    math::{
        Quantity,
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

fn make(
    base: [f64; 3],
    direction: [f64; 3],
    base_radius: f64,
    tip_radius: f64,
    height: f64,
) -> Cone {
    Cone::new(
        point(base),
        axis(direction),
        length(base_radius),
        length(tip_radius),
        length(height),
    )
    .unwrap()
}

#[test]
fn a_frustum_has_its_wall_cap_and_rim_distances() -> Result<(), AssertionError> {
    let cone = make([0.0; 3], [0.0, 0.0, 1.0], 2.0, 1.0, 3.0);
    let distance = |entries| cone.signed_distance(&point(entries));
    let assert = Assert::default();
    let wall = 4.5 / 10.0_f64.sqrt();
    assert.eq_within_tols(distance([0.0, 0.0, 1.5]), &length(-wall))?;
    assert.eq_within_tols(distance([0.0, 0.0, 0.25]), &length(-0.25))?;
    assert.eq_within_tols(distance([0.5, 0.0, -2.0]), &length(2.0))?;
    assert.eq_within_tols(distance([0.5, 0.0, 5.0]), &length(2.0))?;
    assert.eq_within_tols(distance([2.0, 0.0, 4.0]), &length(2.0_f64.sqrt()))
}

#[test]
fn a_pointed_cone_comes_to_a_point() -> Result<(), AssertionError> {
    let cone = make([0.0; 3], [0.0, 0.0, 1.0], 1.0, 0.0, 2.0);
    let assert = Assert::default();
    let apex = cone.project(&point([0.0, 0.0, 3.0])).unwrap();
    assert.eq_within_tols(&apex.point, &point([0.0, 0.0, 2.0]))?;
    assert.eq_within_tols(cone.signed_distance(&point([0.0, 0.0, 3.0])), &length(1.0))?;
    assert.zero_within_tols(&cone.signed_distance(&point([0.0, 0.0, 2.0])))
}

#[test]
fn projects_onto_the_side_with_its_normal() -> Result<(), AssertionError> {
    let cone = make([0.0; 3], [0.0, 0.0, 1.0], 2.0, 1.0, 3.0);
    let found = cone.project(&point([1.0, 0.0, 1.5])).unwrap();
    let assert = Assert::default();
    assert.zero_within_tols(&cone.signed_distance(&found.point))?;
    let size = 10.0_f64.sqrt();
    assert.eq_within_tols(
        &found.normal,
        &Direction::from([3.0 / size, 0.0, 1.0 / size]),
    )
}

#[test]
fn a_cone_that_is_not_one_is_an_error() {
    let make = |base_radius, tip_radius, height, direction| {
        Cone::new(
            point([0.0; 3]),
            axis(direction),
            length(base_radius),
            length(tip_radius),
            length(height),
        )
    };
    let up = [0.0, 0.0, 1.0];
    assert!(make(1.0, 0.5, 1.0, up).is_ok());
    assert!(make(0.0, 0.0, 1.0, up).is_err());
    assert!(make(-1.0, 0.5, 1.0, up).is_err());
    assert!(make(1.0, -0.5, 1.0, up).is_err());
    assert!(make(1.0, 0.5, 0.0, up).is_err());
    assert!(make(1.0, 0.5, 1.0, [0.0; 3]).is_err());
}

type Case = ([f64; 3], [f64; 3], f64, f64, f64);

const CASES: [Case; 6] = [
    ([0.0; 3], [0.0, 0.0, 1.0], 2.0, 1.0, 3.0),
    ([1.0, -1.0, 0.5], [1.0, 1.0, 0.0], 1.5, 0.0, 2.0),
    ([0.0; 3], [1.0, 2.0, 3.0], 0.5, 2.0, 2.0),
    ([0.0; 3], [0.0, 1.0, 0.0], 1.0, 1.0, 1.0),
    ([0.0; 3], [1.0, 0.0, 0.0], 1e-9, 3e-10, 2e-9),
    ([4e8, 0.0, -1e8], [0.0, 1.0, 0.0], 2e8, 5e7, 1e8),
];

fn unit(direction: [f64; 3]) -> Direction<3> {
    axis(direction).normalized()
}

fn around(case: Case) -> ([f64; 3], [f64; 3]) {
    let (base, direction, base_radius, tip_radius, height) = case;
    let end = &point(base) + &(&unit(direction) * length(height));
    let radius = base_radius.max(tip_radius);
    (
        from_fn(|k| base[k].min(end[k].value()) - radius),
        from_fn(|k| base[k].max(end[k].value()) + radius),
    )
}

#[test]
fn satisfies_the_contract_for_any_size_place_and_direction() -> Result<(), AssertionError> {
    for case in CASES {
        let (base, direction, base_radius, tip_radius, height) = case;
        let cone = make(base, direction, base_radius, tip_radius, height);
        let unit = unit(direction);
        let (low, high) = around(case);
        let points = grid(low, high, 7);
        let scale = height.min(base_radius.max(tip_radius));
        let inside = |p: &Coordinate<3>| {
            let (along, across) = local(&point(base), &unit, p);
            let radius = length(base_radius)
                + (length(tip_radius) - length(base_radius)) * (along / length(height)).value();
            along > length(0.0) && along < length(height) && across < radius
        };
        assert_contract(&cone, &points, scale, inside)?;
        assert_lipschitz(&cone, &points, scale)?;
    }
    Ok(())
}

#[test]
fn agrees_with_a_sampling_of_its_surface() -> Result<(), AssertionError> {
    for case in &CASES[..4] {
        let (base, direction, base_radius, tip_radius, height) = *case;
        let cone = make(base, direction, base_radius, tip_radius, height);
        let corner = |distance: f64, along: f64| [length(distance), length(along)];
        let profile = along_polyline(
            &[
                corner(0.0, 0.0),
                corner(base_radius, 0.0),
                corner(tip_radius, height),
                corner(0.0, height),
            ],
            40,
        );
        let angles = 120;
        let samples = revolved(&point(base), &unit(direction), &profile, angles);
        let radius = base_radius.max(tip_radius);
        let longest = base_radius
            .max(tip_radius)
            .max(height.hypot(base_radius - tip_radius));
        let step = longest / 40.0;
        let spacing = 0.55 * step.hypot(radius * TAU / angles as f64);
        let (low, high) = around(*case);
        assert_against_samples(&cone, &grid(low, high, 5), &samples, spacing)?;
    }
    Ok(())
}
