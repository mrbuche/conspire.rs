use super::Cylinder;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::contract::{
            along_polyline, assert_against_samples, assert_contract, assert_lipschitz, grid, local,
            revolved,
        },
        solid::SolidOracle,
    },
    math::{Quantity, Tensor},
    units::Length,
};
use std::{array::from_fn, f64::consts::TAU};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn axis(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

fn upright() -> Cylinder {
    Cylinder::new(
        point([0.0; 3]),
        axis([0.0, 0.0, 1.0]),
        length(2.0),
        length(6.0),
    )
    .unwrap()
}

#[test]
fn signed_distance_is_negative_inside_radial_and_axial() {
    let cylinder = upright();
    let distance = |entries| cylinder.signed_distance(&point(entries));
    assert!((distance([0.0, 0.0, 3.0]) + length(2.0)).abs() < length(1e-12));
    assert!((distance([3.0, 0.0, 3.0]) - length(1.0)).abs() < length(1e-12));
    assert!((distance([0.0, 0.0, 8.0]) - length(2.0)).abs() < length(1e-12));
    assert!((distance([0.0, 0.0, 0.5]) + length(0.5)).abs() < length(1e-12));
    assert!((distance([3.0, 0.0, 7.0]) - length(2.0_f64.sqrt())).abs() < length(1e-12));
}

#[test]
fn projects_onto_the_wall_and_the_caps() {
    let cylinder = upright();
    let wall = cylinder.project(&point([1.9, 0.0, 3.0])).unwrap();
    assert!((&wall.point - &point([2.0, 0.0, 3.0])).norm() < length(1e-12));
    assert!((&wall.normal - &[1.0, 0.0, 0.0].into()).norm() < Quantity::new(1e-12));
    let cap = cylinder.project(&point([0.5, 0.0, 5.8])).unwrap();
    assert!((&cap.point - &point([0.5, 0.0, 6.0])).norm() < length(1e-12));
    assert!((&cap.normal - &[0.0, 0.0, 1.0].into()).norm() < Quantity::new(1e-12));
    let base = cylinder.project(&point([0.5, 0.0, 0.2])).unwrap();
    assert!((&base.normal - &[0.0, 0.0, -1.0].into()).norm() < Quantity::new(1e-12));
}

#[test]
fn on_the_axis_the_projection_is_still_on_the_surface() {
    let cylinder = upright();
    for z in [-1.0, 0.0, 0.3, 3.0, 5.7, 6.0, 7.0] {
        let found = cylinder.project(&point([0.0, 0.0, z])).unwrap();
        assert!(
            cylinder.signed_distance(&found.point).abs() < length(1e-12),
            "{z}"
        );
    }
}

#[test]
fn a_tilted_axis_is_measured_along_itself() {
    let cylinder = Cylinder::new(
        point([0.0; 3]),
        axis([1.0, 1.0, 0.0]),
        length(1.0),
        length(8.0_f64.sqrt()),
    )
    .unwrap();
    let distance = |entries| cylinder.signed_distance(&point(entries));
    assert!((distance([1.0, 1.0, 0.0]) + length(1.0)).abs() < length(1e-9));
    assert!(distance([1.0, 1.0, 1.0]).abs() < length(1e-9));
}

#[test]
fn a_cylinder_that_is_not_one_is_an_error() {
    let make = |axis_entries, radius, height| {
        Cylinder::new(
            point([0.0; 3]),
            axis(axis_entries),
            length(radius),
            length(height),
        )
    };
    assert!(make([0.0, 0.0, 1.0], 0.0, 1.0).is_err());
    assert!(make([0.0, 0.0, 1.0], 1.0, 0.0).is_err());
    assert!(make([0.0, 0.0, 1.0], -1.0, 1.0).is_err());
    assert!(make([0.0; 3], 1.0, 1.0).is_err());
}

const CASES: [([f64; 3], [f64; 3], f64, f64); 5] = [
    ([0.0; 3], [0.0, 0.0, 1.0], 2.0, 6.0),
    ([1.0, -1.0, 0.5], [1.0, 1.0, 0.0], 1.0, 2.5),
    ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 0.7, 1.9),
    ([0.0; 3], [1.0, 0.0, 0.0], 1e-9, 3e-9),
    ([4e8, 0.0, -1e8], [0.0, 1.0, 0.0], 2e8, 1e8),
];

fn make(base: [f64; 3], direction: [f64; 3], radius: f64, height: f64) -> Cylinder {
    Cylinder::new(point(base), axis(direction), length(radius), length(height)).unwrap()
}

fn unit(direction: [f64; 3]) -> Direction<3> {
    axis(direction).normalized()
}

fn around(base: [f64; 3], direction: [f64; 3], radius: f64, height: f64) -> ([f64; 3], [f64; 3]) {
    let end = &point(base) + &(&unit(direction) * length(height));
    (
        from_fn(|k| base[k].min(end[k].value()) - radius),
        from_fn(|k| base[k].max(end[k].value()) + radius),
    )
}

#[test]
fn satisfies_the_contract_for_any_size_place_and_direction() {
    for (base, direction, radius, height) in CASES {
        let cylinder = make(base, direction, radius, height);
        let unit = unit(direction);
        let (low, high) = around(base, direction, radius, height);
        let points = grid(low, high, 7);
        let scale = radius.min(height);
        let inside = |p: &Coordinate<3>| {
            let (along, across) = local(&point(base), &unit, p);
            across < length(radius) && along > length(0.0) && along < length(height)
        };
        assert_contract(&cylinder, &points, scale, inside);
        assert_lipschitz(&cylinder, &points, scale);
    }
}

#[test]
fn agrees_with_a_sampling_of_its_surface() {
    for (base, direction, radius, height) in &CASES[..3] {
        let cylinder = make(*base, *direction, *radius, *height);
        let unit = unit(*direction);
        let corner = |distance: f64, along: f64| [length(distance), length(along)];
        let profile = along_polyline(
            &[
                corner(0.0, 0.0),
                corner(*radius, 0.0),
                corner(*radius, *height),
                corner(0.0, *height),
            ],
            30,
        );
        let angles = 120;
        let samples = revolved(&point(*base), &unit, &profile, angles);
        let step = radius.max(*height) / 30.0;
        let spacing = 0.55 * step.hypot(radius * TAU / angles as f64);
        let (low, high) = around(*base, *direction, *radius, *height);
        assert_against_samples(&cylinder, &grid(low, high, 5), &samples, spacing);
    }
}
