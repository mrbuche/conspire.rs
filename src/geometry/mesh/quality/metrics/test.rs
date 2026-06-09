use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn hex(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    Mesh::from((connectivities, coordinates.into()))
}

fn tet(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

const UNIT_CUBE: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

#[test]
fn unit_cube_is_perfect() {
    let mesh = hex(UNIT_CUBE.to_vec());
    assert_eq!(mesh.jacobians(), vec![1.0]);
    assert_eq!(mesh.scaled_jacobians(), vec![1.0]);
}

#[test]
fn scaled_is_normalized_jacobian_is_volume() {
    let mesh = hex(UNIT_CUBE.map(|point| point.map(|x| 2.0 * x)).to_vec());
    assert_eq!(mesh.jacobians(), vec![8.0]);
    assert_eq!(mesh.scaled_jacobians(), vec![1.0]);
}

#[test]
fn inverted_hex_is_negative() {
    let mut coordinates = UNIT_CUBE.to_vec();
    coordinates[4] = [0.0, 0.0, -1.0];
    let mesh = hex(coordinates);
    assert!(mesh.scaled_jacobians()[0] < 0.0);
    assert!(mesh.jacobians()[0] < 0.0);
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
    assert_eq!(mesh.jacobians(), vec![1.0]);
    let scaled = mesh.scaled_jacobians()[0];
    assert!(scaled > 0.0 && scaled <= 1.0);
}

#[test]
fn regular_tet_is_perfect() {
    let mesh = tet(REGULAR_TET.to_vec());
    assert!((mesh.scaled_jacobians()[0] - 1.0).abs() < 1.0e-12);
    assert!(mesh.jacobians()[0] > 0.0);
}

#[test]
fn inverted_tet_is_negative() {
    let mut coordinates = REGULAR_TET.to_vec();
    coordinates.swap(1, 2);
    let mesh = tet(coordinates);
    assert!(mesh.scaled_jacobians()[0] < 0.0);
    assert!(mesh.jacobians()[0] < 0.0);
}
