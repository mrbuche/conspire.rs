use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn wedge(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Wedge(vec![[0, 1, 2, 3, 4, 5]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

const HALF_SQRT_3: f64 = 0.866_025_403_784_438_6;

/// Equilateral triangular faces, unit height: every edge has unit length.
const PERFECT: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, HALF_SQRT_3, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.5, HALF_SQRT_3, 1.0],
];

#[test]
fn perfect_wedge_is_unit() {
    let mesh = wedge(PERFECT.to_vec());
    assert!((mesh.minimum_scaled_jacobians()[0][0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.minimum_jacobians()[0][0] > 0.0);
    assert!((mesh.maximum_edge_ratios()[0][0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.maximum_skews()[0][0].abs() < 1.0e-12);
    assert!((mesh.volumes()[0][0] - HALF_SQRT_3 / 2.0).abs() < 1.0e-12);
}

#[test]
fn right_wedge_in_range() {
    let mesh = wedge(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    let scaled = mesh.minimum_scaled_jacobians()[0][0];
    assert!(scaled > 0.0 && scaled <= 1.0);
    assert!(mesh.minimum_jacobians()[0][0] > 0.0);
    assert!((mesh.volumes()[0][0] - 0.5).abs() < 1.0e-12);
}

#[test]
fn inverted_wedge_is_negative() {
    let mut coordinates = PERFECT.to_vec();
    coordinates.swap(1, 2);
    let mesh = wedge(coordinates);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}
