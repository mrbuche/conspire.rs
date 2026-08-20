use super::{dunyach_length, graduate, sizing_field};
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
        dunyach_length(curvature(0.0), tolerance, minimum, maximum),
        maximum
    );
    assert!(
        (dunyach_length(curvature(1.0), tolerance, minimum, maximum).value() - 0.57_f64.sqrt())
            .abs()
            < 1.0e-12
    );
    assert_eq!(
        dunyach_length(curvature(100.0), tolerance, minimum, maximum),
        minimum
    );
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
    );
    assert!(
        field
            .iter()
            .all(|&length| (length.value() - 2.0).abs() < 1.0e-9),
        "a fold is piecewise planar, so no tolerance should refine it"
    );
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
    );
    let expected = dunyach_length(Quantity::new(1.0), tolerance, minimum, maximum);
    let interior = field[2 * 9 + 4];
    assert!(
        (interior.value() - expected.value()).abs() < 5.0e-2 * expected.value(),
        "interior size {interior} should follow the curvature to {expected}"
    );
    assert!(interior < maximum);
}
