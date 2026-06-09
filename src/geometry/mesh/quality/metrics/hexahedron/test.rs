use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn hex(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
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
    assert_eq!(mesh.minimum_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![1.0]]);
}

#[test]
fn scaled_is_normalized_jacobian_is_volume() {
    let mesh = hex(UNIT_CUBE.map(|point| point.map(|x| 2.0 * x)).to_vec());
    assert_eq!(mesh.minimum_jacobians(), vec![vec![8.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![1.0]]);
}

#[test]
fn stretched_hex_edge_ratio_is_longest_over_shortest() {
    let mesh = hex(UNIT_CUBE.map(|[x, y, z]| [x, y, 4.0 * z]).to_vec());
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![4.0]]);
}

#[test]
fn inverted_hex_is_negative() {
    let mut coordinates = UNIT_CUBE.to_vec();
    coordinates[4] = [0.0, 0.0, -1.0];
    let mesh = hex(coordinates);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}