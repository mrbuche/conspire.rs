use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn tet(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

const REGULAR_TET: [[f64; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
];

#[test]
fn right_tet_jacobian_is_six_volumes() {
    let mesh = tet(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    assert_eq!(mesh.minimum_jacobians(), vec![vec![1.0]]);
    let scaled = mesh.minimum_scaled_jacobians()[0][0];
    assert!(scaled > 0.0 && scaled <= 1.0);
}

#[test]
fn regular_tet_is_perfect() {
    let mesh = tet(REGULAR_TET.to_vec());
    assert!((mesh.minimum_scaled_jacobians()[0][0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.minimum_jacobians()[0][0] > 0.0);
    assert!((mesh.maximum_edge_ratios()[0][0] - 1.0).abs() < 1.0e-12);
}

#[test]
fn inverted_tet_is_negative() {
    let mut coordinates = REGULAR_TET.to_vec();
    coordinates.swap(1, 2);
    let mesh = tet(coordinates);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}
