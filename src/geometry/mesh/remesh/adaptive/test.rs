use super::{dunyach_length, graduate, sizing_field};
use crate::{geometry::Coordinates, math::Tensor};

// A flat 4x2 ladder: row 0 at y=0, row 1 at y=1, two triangles per cell.
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
    let (tolerance, minimum, maximum) = (0.1, 0.1, 2.0);
    assert_eq!(dunyach_length(0.0, tolerance, minimum, maximum), maximum);
    assert!((dunyach_length(1.0, tolerance, minimum, maximum) - 0.57_f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(dunyach_length(100.0, tolerance, minimum, maximum), minimum);
}

#[test]
fn graduate_enforces_lipschitz() {
    let (connectivity, coordinates) = ladder();
    let gradation = 0.5;
    let mut field = vec![2.0; coordinates.len()];
    field[0] = 0.1;
    graduate(&mut field, &connectivity, &coordinates, gradation);
    for &[a, b, c] in &connectivity {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            let distance = (&coordinates[j] - &coordinates[i]).norm();
            assert!((field[i] - field[j]).abs() <= gradation * distance + 1.0e-9);
        }
    }
    assert!(field[0] < 0.2, "the small seed survives gradation");
}

#[test]
fn sizing_field_is_uniform_on_flat_mesh() {
    let (connectivity, coordinates) = ladder();
    let field = sizing_field(&connectivity, &coordinates, 0.1, 0.1, 2.0, 0.5);
    assert!(field.iter().all(|&length| (length - 2.0).abs() < 1.0e-9));
}
