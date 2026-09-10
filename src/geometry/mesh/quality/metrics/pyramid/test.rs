use crate::geometry::mesh::{Connectivity, Mesh, Verdict};
use std::f64::consts::FRAC_1_SQRT_2;

fn pyramid(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Pyramidal(vec![[0, 1, 2, 3, 4]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

const PERFECT: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.5, 0.5, FRAC_1_SQRT_2],
];

#[test]
fn perfect_pyramid_is_unit() {
    let mesh = pyramid(PERFECT.to_vec());
    assert!((mesh.minimum_scaled_jacobians()[0][0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.minimum_jacobians()[0][0] > 0.0);
    assert!((mesh.maximum_edge_ratios()[0][0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.maximum_skews()[0][0].abs() < 1.0e-12);
    assert!((mesh.volumes()[0][0] - FRAC_1_SQRT_2 / 3.0).abs() < 1.0e-12);
}

#[test]
fn skewed_pyramid_in_range() {
    let mut coordinates = PERFECT.to_vec();
    coordinates[4] = [1.3, 0.5, 0.6];
    let mesh = pyramid(coordinates);
    let scaled = mesh.minimum_scaled_jacobians()[0][0];
    assert!(scaled > 0.0 && scaled <= 1.0);
    assert!(mesh.minimum_jacobians()[0][0] > 0.0);
}

#[test]
fn inverted_pyramid_is_negative() {
    let mut coordinates = PERFECT.to_vec();
    coordinates[4] = [0.5, 0.5, -FRAC_1_SQRT_2];
    let mesh = pyramid(coordinates);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}
