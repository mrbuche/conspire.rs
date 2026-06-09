use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn tri<const D: usize>(coordinates: Vec<[f64; D]>) -> Mesh<D> {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

#[test]
fn equilateral_is_perfect() {
    let mesh = tri(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.75_f64.sqrt()]]);
    assert!((mesh.minimum_scaled_jacobians()[0][0] - 1.0).abs() < 1.0e-10);
    assert!((mesh.maximum_edge_ratios()[0][0] - 1.0).abs() < 1.0e-10);
}

#[test]
fn right_triangle_jacobian_is_twice_area() {
    let mesh = tri(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    assert_eq!(mesh.minimum_jacobians(), vec![vec![1.0]]);
}

#[test]
fn inverted_triangle_is_negative() {
    let mesh = tri(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
}

#[test]
fn embedded_in_three_dimensions_is_unsigned() {
    let mesh = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.75_f64.sqrt(), 0.0]]);
    assert!((mesh.minimum_scaled_jacobians()[0][0] - 1.0).abs() < 1.0e-10);
    let reversed = tri(vec![[0.5, 0.75_f64.sqrt(), 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert!(reversed.minimum_scaled_jacobians()[0][0] > 0.0);
}