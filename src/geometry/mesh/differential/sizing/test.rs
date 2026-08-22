use super::{Unresolved, dunyach_length, graduate, sizing_field};
use crate::{
    geometry::Coordinates,
    math::{Quantity, Tensor},
};

fn ladder() -> (Vec<[usize; 3]>, Coordinates<3>) {
    let mut points = Vec::new();
    for y in 0..2 {
        for x in 0..4 {
            points.push([x as f64, y as f64, 0.0]);
        }
    }
    let mut connectivity = Vec::new();
    for x in 0..3 {
        let (a, b, c, d) = (x, x + 1, x + 4, x + 5);
        connectivity.push([a, b, d]);
        connectivity.push([a, d, c]);
    }
    (connectivity, Coordinates::from(points))
}

#[test]
fn dunyach_length_maps_curvature() {
    let (tolerance, minimum, maximum) =
        (Quantity::new(0.1), Quantity::new(0.1), Quantity::new(2.0));
    let curvature = Quantity::new;
    assert_eq!(
        dunyach_length(
            curvature(0.0),
            tolerance,
            minimum,
            maximum,
            Unresolved::Radius
        ),
        maximum
    );
    assert!(
        (dunyach_length(
            curvature(1.0),
            tolerance,
            minimum,
            maximum,
            Unresolved::Radius
        )
        .value()
            - 0.57_f64.sqrt())
        .abs()
            < 1.0e-12
    );
    assert_eq!(
        dunyach_length(
            curvature(100.0),
            tolerance,
            minimum,
            maximum,
            Unresolved::Radius
        ),
        minimum
    );
}

#[test]
fn dunyach_length_never_shortens_as_tolerance_loosens() {
    let (minimum, maximum) = (Quantity::new(0.1), Quantity::new(2.0));
    let curvature = Quantity::new(1.0);
    let lengths: Vec<_> = [0.05, 0.1, 0.5, 1.0, 1.5, 3.0]
        .into_iter()
        .map(|tolerance| {
            dunyach_length(
                curvature,
                Quantity::new(tolerance),
                minimum,
                maximum,
                Unresolved::Radius,
            )
        })
        .collect();
    lengths
        .windows(2)
        .for_each(|pair| assert!(pair[1] >= pair[0]));
    assert!((lengths[5].value() - 3.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn a_feature_sharper_than_the_tolerance_splits_the_two_arms() {
    let (tolerance, minimum, maximum) = (
        Quantity::new(1.0e-2),
        Quantity::new(1.0e-4),
        Quantity::new(1.0),
    );
    let curvature = Quantity::new(1.0e3);
    let floor = dunyach_length(curvature, tolerance, minimum, maximum, Unresolved::Minimum);
    let radius = dunyach_length(curvature, tolerance, minimum, maximum, Unresolved::Radius);
    assert_eq!(floor, minimum);
    assert!((radius.value() - 3.0_f64.sqrt() * 1.0e-3).abs() < 1.0e-12);
    assert!(radius > floor);
}

#[test]
fn graduate_enforces_lipschitz() {
    let (connectivity, coordinates) = ladder();
    let gradation = 0.5;
    let mut field = vec![Quantity::new(2.0); coordinates.len()];
    field[0] = Quantity::new(0.1);
    graduate(&mut field, &connectivity, &coordinates, gradation);
    for &[a, b, c] in &connectivity {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            let distance = (&coordinates[j] - &coordinates[i]).norm();
            assert!((field[i] - field[j]).abs().value() <= gradation * distance.value() + 1.0e-9);
        }
    }
    assert!(
        field[0] < Quantity::new(0.2),
        "the small seed survives gradation"
    );
}

#[test]
fn sizing_field_is_uniform_on_flat_mesh() {
    let (connectivity, coordinates) = ladder();
    let field = sizing_field(
        &connectivity,
        &coordinates,
        Quantity::new(0.1),
        Quantity::new(0.1),
        Quantity::new(2.0),
        0.5,
        Unresolved::Minimum,
    );
    assert!(
        field
            .iter()
            .all(|&length| (length.value() - 2.0).abs() < 1.0e-9)
    );
}

fn strip(
    columns: usize,
    rows: usize,
    place: impl Fn(usize, usize) -> [f64; 3],
) -> (Vec<[usize; 3]>, Coordinates<3>) {
    let mut points = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            points.push(place(column, row));
        }
    }
    let mut connectivity = Vec::new();
    for row in 0..rows - 1 {
        for column in 0..columns - 1 {
            let a = row * columns + column;
            let (b, c, d) = (a + 1, a + columns, a + columns + 1);
            connectivity.push([a, b, d]);
            connectivity.push([a, d, c]);
        }
    }
    (connectivity, Coordinates::from(points))
}

fn folded() -> (Vec<[usize; 3]>, Coordinates<3>) {
    strip(7, 5, |column, row| {
        let (along, across) = (column as f64 - 3.0, row as f64);
        if along < 0.0 {
            [along, across, 0.0]
        } else {
            [0.0, across, along]
        }
    })
}

fn cylinder() -> (Vec<[usize; 3]>, Coordinates<3>) {
    strip(9, 5, |column, row| {
        let angle = column as f64 * std::f64::consts::FRAC_PI_2 / 8.0;
        [angle.cos(), row as f64 / 4.0, angle.sin()]
    })
}

#[test]
fn sizing_field_ignores_creases() {
    let (connectivity, coordinates) = folded();
    let field = sizing_field(
        &connectivity,
        &coordinates,
        Quantity::new(1.0e-4),
        Quantity::new(1.0e-3),
        Quantity::new(2.0),
        0.5,
        Unresolved::Minimum,
    );
    assert!(
        field
            .iter()
            .all(|&length| (length.value() - 2.0).abs() < 1.0e-9),
        "a fold is piecewise planar, so no tolerance should refine it"
    );
}

fn creased_cylinder() -> (Vec<[usize; 3]>, Coordinates<3>) {
    strip(13, 5, |column, row| {
        let across = row as f64 / 4.0;
        if column <= 8 {
            let angle = column as f64 * std::f64::consts::FRAC_PI_2 / 8.0;
            [angle.cos(), across, angle.sin()]
        } else {
            [(column - 8) as f64 / 8.0, across, 1.0]
        }
    })
}

#[test]
fn a_crease_inherits_the_curvature_around_it() {
    let (connectivity, coordinates) = creased_cylinder();
    let maximum = Quantity::new(2.0);
    let field = sizing_field(
        &connectivity,
        &coordinates,
        Quantity::new(1.0e-3),
        Quantity::new(1.0e-4),
        maximum,
        1.0e3,
        Unresolved::Minimum,
    );
    let (crease, curved) = (field[2 * 13 + 8], field[2 * 13 + 7]);
    assert!(
        crease <= curved,
        "crease size {crease} should not exceed the curved size {curved} beside it"
    );
    assert!(crease < maximum * 0.1);
}

#[test]
fn sizing_field_still_follows_smooth_curvature() {
    let (connectivity, coordinates) = cylinder();
    let (tolerance, minimum, maximum) = (
        Quantity::new(1.0e-3),
        Quantity::new(1.0e-4),
        Quantity::new(2.0),
    );
    let field = sizing_field(
        &connectivity,
        &coordinates,
        tolerance,
        minimum,
        maximum,
        0.5,
        Unresolved::Minimum,
    );
    let expected = dunyach_length(
        Quantity::new(1.0),
        tolerance,
        minimum,
        maximum,
        Unresolved::Minimum,
    );
    let interior = field[2 * 9 + 4];
    assert!(
        (interior.value() - expected.value()).abs() < 5.0e-2 * expected.value(),
        "interior size {interior} should follow the curvature to {expected}"
    );
    assert!(interior < maximum);
}
